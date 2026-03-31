"""
RunPod Serverless handler for SAM 3 image segmentation.

Accepts a job with the following input fields:
  - image        (str, required): Base64-encoded image OR a public URL.
  - text_prompt  (str, optional): Text description of objects to segment. Default: "object".
  - box_prompt   (list, optional): Bounding box [x1, y1, x2, y2] in pixel coordinates.
  - confidence_threshold (float, optional): Min confidence 0.0-1.0. Default: 0.5.
  - mask_opacity (float, optional): Overlay opacity 0.0-1.0. Default: 0.5.
  - mask_color   (str, optional): One of green/red/blue/yellow/cyan/magenta. Default: "green".
  - output_format (str, optional): "png" (base64 annotated image) or "json" (structured). Default: "png".
"""

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
    """Load an image from a base64 string or URL."""
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
    """Encode a binary mask as run-length encoding."""
    if mask.ndim > 2:
        mask = mask.squeeze()
    flat = (mask > 0.5).flatten().astype(np.uint8)

    if len(flat) == 0:
        return {"counts": [], "size": list(mask.shape)}

    # find positions where the value changes
    changes = np.where(np.diff(flat))[0] + 1
    # include boundaries (0 and len) to capture first and last runs
    positions = np.concatenate([[0], changes, [len(flat)]])
    lengths = np.diff(positions)

    # counts alternate starting with background (0).
    # if the mask starts with foreground (1), prepend a 0-length bg run.
    if flat[0] == 1:
        counts = [0] + lengths.tolist()
    else:
        counts = lengths.tolist()

    return {"counts": counts, "size": list(mask.shape)}


def build_visualization(pil_image, masks, boxes, scores, mask_opacity=0.5, mask_color="green"):
    """Render annotated image with colored mask overlays."""
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
                mask = cv2.resize(
                    mask.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
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

            score = float(scores[i]) if scores.ndim > 0 else 0.0
            label = f"{score:.2f}"
            (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                overlay, (x1, y1 - th_text - 6),
                (x1 + tw + 4, y1), color_rgb.tolist(), -1,
            )
            cv2.putText(
                overlay, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

    return Image.fromarray(overlay)


# ---------------------------------------------------------------------------
# Model loading (runs once at container start)
# ---------------------------------------------------------------------------
print("Initializing SAM 3 model...")
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

if not os.path.exists(MODEL_PATH):
    print("Weights not found locally, downloading...")
    download_weights(MODEL_URL, MODEL_PATH)

from transformers import Sam3Model, Sam3Processor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

print(f"Loading SAM 3 on {DEVICE} with {DTYPE}...")
MODEL = Sam3Model.from_pretrained(MODEL_PATH).to(DEVICE, dtype=DTYPE).eval()
PROCESSOR = Sam3Processor.from_pretrained(MODEL_PATH)
print("SAM 3 model loaded successfully.")


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(job):
    """Process a single RunPod serverless job."""
    job_input = job["input"]

    # --- Parse inputs ---
    image_input = job_input.get("image")
    if not image_input:
        return {"error": "Missing required field: 'image' (base64 string or URL)"}

    text_prompt = job_input.get("text_prompt", "object")
    box_prompt = job_input.get("box_prompt", None)
    confidence_threshold = float(job_input.get("confidence_threshold", 0.5))
    mask_opacity = float(job_input.get("mask_opacity", 0.5))
    mask_color = job_input.get("mask_color", "green")
    output_format = job_input.get("output_format", "png")

    # --- Load image ---
    try:
        pil_image = load_image(image_input)
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}

    width, height = pil_image.size

    # --- Build processor inputs ---
    if box_prompt is not None:
        if isinstance(box_prompt, str):
            box_prompt = json.loads(box_prompt.strip())
        if not isinstance(box_prompt, list) or len(box_prompt) != 4:
            return {"error": "box_prompt must be a list of 4 numbers: [x1, y1, x2, y2]"}
        inputs = PROCESSOR(
            images=pil_image,
            text=text_prompt,
            input_boxes=[[box_prompt]],
            return_tensors="pt",
        ).to(DEVICE, dtype=DTYPE)
    else:
        inputs = PROCESSOR(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt",
        ).to(DEVICE, dtype=DTYPE)

    # --- Inference ---
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

    # --- Build output ---
    if output_format == "json":
        detections = []
        num = int(len(scores_np)) if scores_np.ndim > 0 else 0
        for i in range(num):
            det = {
                "score": float(scores_np[i]),
                "box": boxes_np[i].tolist() if boxes_np.ndim > 1 else [],
            }
            if masks_np.ndim > 2:
                det["mask_rle"] = mask_to_rle(masks_np[i])
            detections.append(det)
        return {"num_detections": num, "detections": detections}
    else:
        result_image = build_visualization(
            pil_image, masks_np, boxes_np, scores_np,
            mask_opacity=mask_opacity,
            mask_color=mask_color,
        )
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image_base64": b64_image}


runpod.serverless.start({"handler": handler})
