# Prediction interface for Cog ⚙️
# https://cog.run/python
#
# SAM 3 image segmentation wrapper.
# Supports text-prompted and box-prompted segmentation.

import json
import os
import subprocess
import tempfile
import time
from typing import Optional

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image

MODEL_PATH = "/src/checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/facebook/sam3/model.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the SAM 3 model into memory."""
        from transformers import Sam3Model, Sam3Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        # Debug: check what's actually at the model path
        print(f"Checking MODEL_PATH: {MODEL_PATH}")
        print(f"  exists: {os.path.exists(MODEL_PATH)}")
        if os.path.exists(MODEL_PATH):
            print(f"  contents: {os.listdir(MODEL_PATH)}")
        else:
            # Also check if it landed somewhere else
            for check in ["/src", "/src/checkpoints", "/src/model"]:
                if os.path.exists(check):
                    print(f"  {check} exists, contents: {os.listdir(check)}")

        # Download weights only if not baked into the image
        if not os.path.exists(MODEL_PATH):
            print("Weights not found, downloading...")
            download_weights(MODEL_URL, MODEL_PATH)
            print(f"  after download: {os.listdir(MODEL_PATH)}")

        print(f"Loading SAM 3 model on {self.device} with {self.dtype}...")
        self.model = (
            Sam3Model.from_pretrained(MODEL_PATH)
            .to(self.device, dtype=self.dtype)
            .eval()
        )
        self.processor = Sam3Processor.from_pretrained(MODEL_PATH)
        print("SAM 3 model loaded successfully.")

    def predict(
        self,
        image: Path = Input(description="Input image to segment."),
        text_prompt: str = Input(
            default="object",
            description=(
                "Text description of the object(s) to segment "
                '(e.g. "yellow school bus", "person wearing red").'
            ),
        ),
        box_prompt: Optional[str] = Input(
            default=None,
            description=(
                "Optional: bounding box as JSON [x1, y1, x2, y2] in pixel coordinates. "
                'Example: "[100, 200, 400, 500]".'
            ),
        ),
        confidence_threshold: float = Input(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Minimum confidence score to include a mask in output.",
        ),
        mask_opacity: float = Input(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Opacity of the mask overlay on the visualization.",
        ),
        mask_color: str = Input(
            default="green",
            choices=["green", "red", "blue", "yellow", "cyan", "magenta"],
            description="Color of the mask overlay.",
        ),
        output_format: str = Input(
            default="png",
            choices=["png", "json"],
            description=(
                "'png' returns annotated image; "
                "'json' returns masks, boxes, and scores."
            ),
        ),
    ) -> Path:
        """Run SAM 3 segmentation on an input image."""
        pil_image = Image.open(str(image)).convert("RGB")
        width, height = pil_image.size

        # Prepare inputs via the processor
        if box_prompt is not None:
            box = json.loads(box_prompt.strip())
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(
                    "box_prompt must be a JSON list of 4 numbers: [x1, y1, x2, y2]"
                )
            inputs = self.processor(
                images=pil_image,
                text=text_prompt,
                input_boxes=[[box]],
                return_tensors="pt",
            ).to(self.device, dtype=self.dtype)
        else:
            inputs = self.processor(
                images=pil_image,
                text=text_prompt,
                return_tensors="pt",
            ).to(self.device, dtype=self.dtype)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get masks, boxes, and scores
        results = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(height, width)],
            threshold=confidence_threshold,
        )[0]

        masks = results.get("masks", None)
        scores = results.get("scores", None)
        boxes = results.get("boxes", None)

        # Convert to numpy
        masks_np = self._to_numpy(masks)
        scores_np = self._to_numpy(scores)
        boxes_np = self._to_numpy(boxes)

        # Note: threshold is already applied inside post_process_instance_segmentation,
        # so no additional filtering is needed here.

        if output_format == "json":
            return self._output_json(masks_np, boxes_np, scores_np)
        else:
            return self._output_visualization(
                pil_image, masks_np, boxes_np, scores_np,
                mask_opacity=mask_opacity,
                mask_color=mask_color,
            )

    @staticmethod
    def _to_numpy(tensor):
        if tensor is None:
            return np.empty((0,))
        if torch.is_tensor(tensor):
            return tensor.cpu().float().numpy()
        if isinstance(tensor, list):
            if len(tensor) > 0 and torch.is_tensor(tensor[0]):
                return torch.stack(tensor).cpu().float().numpy()
            return np.array(tensor) if len(tensor) > 0 else np.empty((0,))
        return np.array(tensor)

    def _output_json(self, masks, boxes, scores) -> Path:
        """Return detections as a JSON file."""
        detections = []
        num = int(len(scores)) if scores.ndim > 0 else 0
        for i in range(num):
            det = {
                "score": float(scores[i]),
                "box": boxes[i].tolist() if boxes.ndim > 1 else [],
            }
            if masks.ndim > 2:
                det["mask_rle"] = self._mask_to_rle(masks[i])
            detections.append(det)

        result = {"num_detections": num, "detections": detections}
        output_path = tempfile.mktemp(suffix=".json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        return Path(output_path)

    def _output_visualization(
        self, pil_image, masks, boxes, scores,
        mask_opacity=0.5, mask_color="green",
    ) -> Path:
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
        color_rgb = np.array(
            colors.get(mask_color.lower(), [0, 255, 0]), dtype=np.uint8
        )

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

                # Color overlay
                overlay[mask_bool] = (
                    overlay[mask_bool] * (1 - mask_opacity)
                    + color_rgb * mask_opacity
                ).astype(np.uint8)

                # Contour
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(overlay, contours, -1, color_rgb.tolist(), 2)

            # Draw box
            if boxes.ndim > 1 and len(boxes[i]) == 4:
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color_rgb.tolist(), 2)

                score = float(scores[i]) if scores.ndim > 0 else 0.0
                label = f"{score:.2f}"
                (tw, th_text), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    overlay, (x1, y1 - th_text - 6),
                    (x1 + tw + 4, y1), color_rgb.tolist(), -1,
                )
                cv2.putText(
                    overlay, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

        output_path = tempfile.mktemp(suffix=".png")
        Image.fromarray(overlay).save(output_path)
        return Path(output_path)

    @staticmethod
    def _mask_to_rle(mask: np.ndarray) -> dict:
        """Encode a binary mask as run-length encoding."""
        if mask.ndim > 2:
            mask = mask.squeeze()
        flat = (mask > 0.5).flatten().astype(np.uint8)
        diff = np.diff(flat, prepend=0, append=0)
        starts = np.where(diff != 0)[0]
        lengths = np.diff(starts)
        counts = [0] + lengths.tolist() if flat[0] == 1 else lengths.tolist()
        return {"counts": counts, "size": list(mask.shape)}
