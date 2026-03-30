#!/usr/bin/env python3
"""
Download SAM 3 model weights from HuggingFace.

Run this BEFORE `cog build` so the weights are available locally:

    1. Request access at https://huggingface.co/facebook/sam3
    2. Generate a HuggingFace access token at https://huggingface.co/settings/tokens
    3. Run:
         export HF_TOKEN=hf_your_token_here
         python download_weights.py

This downloads sam3.pt into ./weights/ which cog.yaml copies into the image.
"""

import os
import sys
from pathlib import Path

def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print()
        print("  1. Request access at https://huggingface.co/facebook/sam3")
        print("  2. Create a token at https://huggingface.co/settings/tokens")
        print("  3. Export it:  export HF_TOKEN=hf_your_token_here")
        sys.exit(1)

    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)

    output_path = weights_dir / "sam3.pt"
    if output_path.exists():
        print(f"Weights already exist at {output_path} — skipping download.")
        print("Delete the file and re-run to force a fresh download.")
        return

    print("Downloading SAM 3 checkpoint from facebook/sam3 ...")
    downloaded = hf_hub_download(
        repo_id="facebook/sam3",
        filename="sam3.pt",
        local_dir=str(weights_dir),
        token=token,
    )
    print(f"Downloaded to: {downloaded}")

    # Verify the file landed in the expected place
    if output_path.exists():
        size_gb = output_path.stat().st_size / (1024 ** 3)
        print(f"Success! Weights saved to {output_path} ({size_gb:.2f} GB)")
    else:
        # huggingface_hub may use a different local structure; move if needed
        dl_path = Path(downloaded)
        if dl_path.exists() and dl_path != output_path:
            dl_path.rename(output_path)
            print(f"Moved to {output_path}")
        print("Done.")


if __name__ == "__main__":
    main()
