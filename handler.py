import base64
import io
import json
import os
import subprocess
import time
from typing import Optional

import cv2
import numpy as np
import runpod
import torch
from PIL import Image

MODEL_PATH = "/src/checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/facebook/sam3/model.tar"


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print(f"Downloading weights from {url} to {dest}")
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print(f"Download completed in {time.time() - start:.1f}s")


def load_image(image_input: str) -> Image.Image:
    if image_input.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(image_input) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    else:
        image_bytes = base64.b64decode(image_input)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def to_numpy(tensor):
    if tensor is None:
        return np.empty((0,))
    if torch.is_tensor(tensor):
        return tensor.cpu().float().numpy()
    if isinstance(tensor, list):
        if len(tensor) > 0 and torch.is_tensor(tensor[0]):
            return torch.stack(tensor).cpu().float().numpy()
        return np.array(tensor) if len(tensor) > 0 else np.empty((0,))
    return np.array(tensor)


def mask_to_rle(mask: np.ndarray) -> dict:
    if mask.ndim > 2:
        mask = mask.squeeze()

    flat = (mask > 0.5).astype(np.uint8).flatten(order="F")

    if flat.size == 0:
        return {"counts": [], "size": list(mask.shape)}

    changes = np.where(np.diff(flat))[0] + 1
    positions = np.concatenate([[0], changes, [len(flat)]])
    lengths = np.diff(positions)

    if flat[0] == 1:
        counts = [0] + lengths.tolist()
    else:
        counts = lengths.tolist()

    return {"counts": counts, "size": list(mask.shape)}


def build_visualization(pil_image, masks, boxes, scores, mask_opacity=0.5, mask_color="green"):
    img = np.array(pil_image)
    overlay = img.copy()
    h, w = img.shape[:2]

    colors = {
        "green": [0, 255, 0],
        "red": [255, 0, 0],
        "blue": [0, 0, 255],
        "yellow": [255, 255, 0],
        "cyan": [0, 255, 255],
        "magenta": [255, 0, 255],
    }
    color_rgb = np.array(colors.get(mask_color.lower(), [0, 255, 0]), dtype=np.uint8)

    num = int(len(scores)) if scores.ndim > 0 else 0
    for i in range(num):
        if masks.ndim > 2:
            mask = masks[i]
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            elif mask.ndim > 2:
                mask = mask.squeeze()

            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

            mask_bool = mask > 0.5

            overlay[mask_bool] = (
                overlay[mask_bool] * (1 - mask_opacity) + color_rgb * mask_opacity
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, color_rgb.tolist(), 2)

        if boxes.ndim > 1 and len(boxes[i]) == 4:
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_rgb.tolist(), 2)

    return Image.fromarray(overlay)


# ---------------- MODEL LOAD ----------------
print("Initializing SAM 3 model...")

if not os.path.exists(MODEL_PATH):
    download_weights(MODEL_URL, MODEL_PATH)

from transformers import Sam3Model, Sam3Processor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

MODEL = Sam3Model.from_pretrained(MODEL_PATH).to(DEVICE, dtype=DTYPE).eval()
PROCESSOR = Sam3Processor.from_pretrained(MODEL_PATH)

print("Model ready")


# ---------------- HANDLER ----------------
def handler(job):
    job_input = job["input"]

    image_input = job_input.get("image")
    if not image_input:
        return {"error": "Missing image"}

    text_prompt = job_input.get("text_prompt", "person")
    confidence_threshold = float(job_input.get("confidence_threshold", 0.5))
    output_format = job_input.get("output_format", "png")

    selected_indices = job_input.get("selected_indices", None)
    return_mask_only = job_input.get("return_mask_only", False)

    # Load image
    try:
        pil_image = load_image(image_input)
    except Exception as e:
        return {"error": str(e)}

    width, height = pil_image.size

    # Prepare input
    inputs = PROCESSOR(
        images=pil_image,
        text=text_prompt,
        return_tensors="pt",
    ).to(DEVICE, dtype=DTYPE)

    with torch.no_grad():
        outputs = MODEL(**inputs)

    results = PROCESSOR.post_process_instance_segmentation(
        outputs,
        target_sizes=[(height, width)],
        threshold=confidence_threshold,
    )[0]

    masks_np = to_numpy(results.get("masks", None))
    scores_np = to_numpy(results.get("scores", None))
    boxes_np = to_numpy(results.get("boxes", None))

    # ---------------- NEW: MASK MERGE ----------------
    if selected_indices is not None:
        try:
            if isinstance(selected_indices, str):
                selected_indices = json.loads(selected_indices)

            selected_indices = [
                i for i in selected_indices if 0 <= i < len(masks_np)
            ]

            if len(selected_indices) == 0:
                return {"error": "No valid selected indices"}

            selected_masks = masks_np[selected_indices]

            combined_mask = np.any(selected_masks > 0.5, axis=0).astype(np.uint8)

            if combined_mask.shape != (height, width):
                combined_mask = cv2.resize(
                    combined_mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

            # expand mask slightly
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

            if return_mask_only:
                bw = (combined_mask * 255).astype(np.uint8)
                img = Image.fromarray(bw)

                buf = io.BytesIO()
                img.save(buf, format="PNG")

                return {
                    "mask_base64": base64.b64encode(buf.getvalue()).decode(),
                    "num_selected": len(selected_indices)
                }

        except Exception as e:
            return {"error": str(e)}

    # ---------------- DEFAULT OUTPUT ----------------
    if output_format == "json":
        detections = []
        for i in range(len(scores_np)):
            det = {
                "score": float(scores_np[i]),
                "box": boxes_np[i].tolist(),
                "mask_rle": mask_to_rle(masks_np[i])
            }
            detections.append(det)

        return {"detections": detections}

    else:
        result_image = build_visualization(
            pil_image, masks_np, boxes_np, scores_np
        )

        buf = io.BytesIO()
        result_image.save(buf, format="PNG")

        return {
            "image_base64": base64.b64encode(buf.getvalue()).decode()
        }


runpod.serverless.start({"handler": handler})
