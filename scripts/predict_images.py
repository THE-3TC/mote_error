#!/usr/bin/env python3
"""
Run a layout model over a directory of unlabelled page images and save predictions to CSV.

Unlike export_predictions.py, no ground truth is required. The output uses the same
columns as export_predictions.py (with source="pred") so it can be used by the same
downstream tooling.

Usage:
    python scripts/predict_images.py --model yolo --images path/to/pages
    python scripts/predict_images.py --model heron --images path/to/pages --batch-size 8
    python scripts/predict_images.py --model ppdoc-l --images path/to/pages

The ppdoc-* models need PaddlePaddle + paddleocr instead of torch; keep them in a
separate environment from yolo/heron (see README).
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

COLUMNS = [
    "filename",
    "image_path",
    "image_width",
    "image_height",
    "model",
    "source",
    "x",
    "y",
    "width",
    "height",
    "class",
    "confidence",
    "ssu_id",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run a layout model over a directory of images (no ground truth needed)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolo", "heron", "ppdoc-l", "ppdoc-m", "ppdoc-s"],
    )
    parser.add_argument("--images", type=str, required=True, help="Directory of page images")
    parser.add_argument(
        "--recursive", action="store_true", help="Search subdirectories for images too"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/{images_dir_name}_{model}_predictions.csv)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per GPU call")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold (default: model default, 0.6 for heron, 0.2 otherwise)",
    )
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        logger.error(f"Images directory does not exist: {images_dir}")
        sys.exit(1)

    pattern = "**/*" if args.recursive else "*"
    image_paths = sorted(
        p for p in images_dir.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        logger.error(f"No images found in {images_dir}")
        sys.exit(1)
    logger.info(f"Found {len(image_paths)} images")

    if args.output is None:
        output_path = (
            Path("results") / f"{images_dir.name}_{args.model.replace('-', '_')}_predictions.csv"
        )
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "yolo":
        from models.doclayout_yolo import DocLayoutYOLO

        kwargs = {} if args.threshold is None else {"conf_threshold": args.threshold}
        model = DocLayoutYOLO(device=args.device, **kwargs)
        model_name = "DocLayout-YOLO"
    elif args.model == "heron":
        from models.docling_heron import DoclingLayoutHeron

        kwargs = {} if args.threshold is None else {"threshold": args.threshold}
        model = DoclingLayoutHeron(device=args.device, **kwargs)
        model_name = "DoclingLayoutHeron"
    else:
        from models.pp_doclayout import PPDocLayout

        model_name = f"PPDocLayout-{args.model.split('-')[1].upper()}"
        kwargs = {} if args.threshold is None else {"conf_threshold": args.threshold}
        model = PPDocLayout(
            model_name=model_name.replace("PPDocLayout", "PP-DocLayout"),
            device=args.device,
            **kwargs,
        )

    logger.info(f"Loading model: {model_name}")
    model.load()

    rows = []
    for start in tqdm(range(0, len(image_paths), args.batch_size), desc="Predicting"):
        batch_paths = image_paths[start : start + args.batch_size]
        batch_preds = model.predict_batch(batch_paths, batch_size=args.batch_size)

        for image_path, predictions in zip(batch_paths, batch_preds):
            with Image.open(image_path) as img:
                image_width, image_height = img.size

            base = {
                "filename": str(image_path.relative_to(images_dir)),
                "image_path": str(image_path),
                "image_width": image_width,
                "image_height": image_height,
                "model": model_name,
                "source": "pred",
                "ssu_id": None,
            }
            for pred in predictions:
                rows.append(
                    {
                        **base,
                        "x": pred["x"],
                        "y": pred["y"],
                        "width": pred["width"],
                        "height": pred["height"],
                        "class": pred["class"],
                        "confidence": pred.get("confidence", None),
                    }
                )

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} prediction boxes for {len(image_paths)} images to {output_path}")


if __name__ == "__main__":
    main()
