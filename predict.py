"""
Cog predictor for Meta SAM 3 (Segment Anything Model 3).

Supports:
  - Text-prompted segmentation: provide a text description to segment matching objects
  - Box-prompted segmentation: provide a bounding box [cx, cy, w, h] normalized to [0,1]
  - Automatic segmentation: segment everything in the image without prompts

Outputs a JSON result with masks, bounding boxes, and confidence scores,
along with an annotated visualization image.
"""

import json
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path as CogPath
from PIL import Image


class Predictor(BasePredictor):
    """Cog predictor wrapping Meta SAM 3 for image segmentation."""

    def setup(self) -> None:
        """Load the SAM 3 model into memory on GPU."""
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Weights are pre-downloaded into the image at build time.
        # See download_weights.py and cog.yaml for details.
        WEIGHTS_PATH = "/src/weights/sam3.pt"

        print(f"Loading SAM 3 model on {self.device}...")
        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=WEIGHTS_PATH,
            load_from_HF=False,
            enable_segmentation=True,
        )
        self.processor = Sam3Processor(
            model=self.model,
            device=self.device,
        )
        print("SAM 3 model loaded successfully.")

    def predict(
        self,
        image: CogPath = Input(description="Input image to segment."),
        text_prompt: str = Input(
            default="",
            description=(
                "Text description of the object(s) to segment "
                '(e.g. "yellow school bus", "person wearing red"). '
                "Leave empty for automatic segmentation of all objects."
            ),
        ),
        box_prompt: str = Input(
            default="",
            description=(
                "Bounding box prompt as JSON: [center_x, center_y, width, height] "
                "with values normalized to [0, 1]. "
                'Example: "[0.5, 0.5, 0.3, 0.4]". '
                "Leave empty to skip box prompting."
            ),
        ),
        confidence_threshold: float = Input(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Minimum confidence score to include a mask in the output.",
        ),
        output_format: str = Input(
            default="png",
            choices=["png", "json"],
            description=(
                "Output format: 'png' returns an annotated visualization image; "
                "'json' returns raw masks, boxes, and scores as a JSON file."
            ),
        ),
    ) -> CogPath:
        """Run SAM 3 segmentation on an input image."""
        # Load the input image
        pil_image = Image.open(str(image)).convert("RGB")

        # Set confidence threshold
        self.processor.set_confidence_threshold(confidence_threshold)

        # Encode the image
        state = self.processor.set_image(pil_image)

        # Run the appropriate inference mode
        if text_prompt.strip():
            state = self.processor.set_text_prompt(
                prompt=text_prompt.strip(),
                state=state,
            )
        elif box_prompt.strip():
            box = json.loads(box_prompt.strip())
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(
                    "box_prompt must be a JSON list of 4 floats: "
                    "[center_x, center_y, width, height]"
                )
            state = self.processor.add_geometric_prompt(
                box=[float(v) for v in box],
                label=True,
                state=state,
            )
        else:
            # No prompt provided: use a generic text prompt to detect all objects.
            # SAM 3 is concept-based; use a broad prompt for general segmentation.
            state = self.processor.set_text_prompt(
                prompt="object",
                state=state,
            )

        # Extract results from state
        masks = state.get("masks")
        boxes = state.get("boxes")
        scores = state.get("scores")

        if masks is None:
            masks = []
            boxes = []
            scores = []

        # Convert tensors to numpy for processing
        if torch.is_tensor(masks):
            masks_np = masks.cpu().numpy()
        elif isinstance(masks, list) and len(masks) > 0 and torch.is_tensor(masks[0]):
            masks_np = torch.stack(masks).cpu().numpy()
        else:
            masks_np = np.array(masks) if len(masks) > 0 else np.empty((0,))

        if torch.is_tensor(boxes):
            boxes_np = boxes.cpu().numpy()
        elif isinstance(boxes, list) and len(boxes) > 0 and torch.is_tensor(boxes[0]):
            boxes_np = torch.stack(boxes).cpu().numpy()
        else:
            boxes_np = np.array(boxes) if len(boxes) > 0 else np.empty((0,))

        if torch.is_tensor(scores):
            scores_np = scores.cpu().numpy()
        elif isinstance(scores, list) and len(scores) > 0 and torch.is_tensor(scores[0]):
            scores_np = torch.stack(scores).cpu().numpy()
        else:
            scores_np = np.array(scores) if len(scores) > 0 else np.empty((0,))

        # Build output
        if output_format == "json":
            return self._output_json(masks_np, boxes_np, scores_np)
        else:
            return self._output_visualization(pil_image, masks_np, boxes_np, scores_np)

    def _output_json(
        self,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> CogPath:
        """Save masks, boxes, and scores as a JSON file."""
        results = {
            "num_detections": int(len(scores)) if scores.ndim > 0 else 0,
            "detections": [],
        }

        num = results["num_detections"]
        for i in range(num):
            detection = {
                "score": float(scores[i]) if scores.ndim > 0 else 0.0,
                "box": boxes[i].tolist() if boxes.ndim > 1 else [],
                "mask_shape": list(masks[i].shape) if masks.ndim > 2 else [],
                "mask_rle": self._mask_to_rle(masks[i]) if masks.ndim > 2 else None,
            }
            results["detections"].append(detection)

        output_path = Path(tempfile.mktemp(suffix=".json"))
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        return CogPath(output_path)

    def _output_visualization(
        self,
        pil_image: Image.Image,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> CogPath:
        """Render an annotated image with mask overlays and bounding boxes."""
        img = np.array(pil_image)
        overlay = img.copy()
        h, w = img.shape[:2]

        num = int(len(scores)) if scores.ndim > 0 else 0
        # Generate distinct colors for each detection
        colors = self._generate_colors(num)

        for i in range(num):
            color = colors[i]

            # Draw mask overlay
            if masks.ndim > 2:
                mask = masks[i]
                # Resize mask to image dimensions if needed
                if mask.shape[-2:] != (h, w):
                    mask_resized = cv2.resize(
                        mask.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    mask_bool = mask_resized > 0.5
                else:
                    mask_bool = mask > 0.5

                # Ensure mask is 2D
                if mask_bool.ndim > 2:
                    mask_bool = mask_bool.squeeze()

                colored_mask = np.zeros_like(img)
                colored_mask[mask_bool] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)

                # Draw mask contour
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(overlay, contours, -1, color, 2)

            # Draw bounding box
            if boxes.ndim > 1 and len(boxes[i]) == 4:
                cx, cy, bw, bh = boxes[i]
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                # Draw score label
                score = float(scores[i]) if scores.ndim > 0 else 0.0
                label = f"{score:.2f}"
                font_scale = 0.5
                thickness = 1
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                cv2.rectangle(
                    overlay,
                    (x1, y1 - th - 6),
                    (x1 + tw + 4, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    overlay,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        output_path = Path(tempfile.mktemp(suffix=".png"))
        result_image = Image.fromarray(overlay)
        result_image.save(str(output_path))
        return CogPath(output_path)

    @staticmethod
    def _generate_colors(n: int) -> list:
        """Generate n visually distinct colors using HSV spacing."""
        colors = []
        for i in range(max(n, 1)):
            hue = int(180 * i / max(n, 1))
            hsv = np.array([[[hue, 255, 200]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
            colors.append(tuple(int(c) for c in rgb))
        return colors

    @staticmethod
    def _mask_to_rle(mask: np.ndarray) -> dict:
        """Encode a binary mask as run-length encoding (RLE)."""
        if mask.ndim > 2:
            mask = mask.squeeze()
        flat = mask.flatten().astype(np.uint8)
        # Compute run lengths
        diff = np.diff(flat, prepend=0, append=0)
        starts = np.where(diff != 0)[0]
        lengths = np.diff(starts)
        # Determine if first run is 0 or 1
        if flat[0] == 1:
            counts = [0] + lengths.tolist()
        else:
            counts = lengths.tolist()
        return {
            "counts": counts,
            "size": list(mask.shape),
        }
