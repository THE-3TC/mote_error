"""Re-score saved Spiritualist predictions with pymote's BenchmarkRunner (no inference).

GT is SpACER's gt_line_boxes.csv: one box per text line, labelled with its article-level SSU
(the same labels as gt_ssu_bboxes.csv and characters_inferred.parquet).

Usage (from the pymote repo root):
    python rescore_spiritualist.py IMAGES_DIR GT_LINES_CSV BBOX_DIR OUTPUT_DIR
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from benchmarks.runner import BenchmarkRunner  # noqa: E402

MODELS = {
    "DocLayout-YOLO": "yolo",
    "DoclingLayoutHeron": "heron",
    "PPDocLayout-L": "ppdoc_l",
    "PPDocLayout-M": "ppdoc_m",
    "PPDocLayout-S": "ppdoc_s",
}


class ReplayModel:
    """Stands in for a layout model, returning boxes saved by an earlier run."""

    def __init__(self, name, csv_path):
        self.model_name = name
        self.model = True  # runner calls load() only when this is None
        df = pd.read_csv(csv_path)
        df = df[df.source == "pred"]
        cols = ["x", "y", "width", "height", "class", "confidence"]
        self.by_file = {f: g[cols].to_dict("records") for f, g in df.groupby("filename")}

    def predict(self, path):
        return self.by_file.get(Path(path).name, [])

    def predict_batch(self, paths, batch_size=None):
        return [self.predict(p) for p in paths]


def line_dataset(images_dir, gt_csv):
    """Samples in the SpiritualistDataset format, with GT lines from the CSV."""
    gt = pd.read_csv(gt_csv)
    samples = []
    for filename, page in gt.groupby("filename", sort=True):
        image_path = Path(images_dir) / filename
        if not image_path.exists():
            continue
        ssu_to_int = {s: i + 1 for i, s in enumerate(page.ssu_id.unique())}
        annotations = [
            {"x": float(r.x), "y": float(r.y), "width": float(r.width), "height": float(r.height),
             "class": "text", "ssu_id": ssu_to_int[r.ssu_id], "ssu_class": "object",
             "confidence": 1.0, "page_id": r.page_id}
            for r in page.itertuples()
        ]
        samples.append({"image_path": str(image_path), "annotations": annotations, "filename": filename})
    return samples


def main(images_dir, gt_csv, bbox_dir, output_dir):
    output_dir = Path(output_dir)
    runner = BenchmarkRunner(Path(images_dir), output_dir, dataset_name="spiritualist")
    runner._dataset = line_dataset(images_dir, gt_csv)  # skips the ALTO XML loader
    print(f"{len(runner._dataset)} pages")
    all_results = {"timestamp": datetime.now().isoformat(), "models": {}}
    for name, key in MODELS.items():
        model = ReplayModel(name, Path(bbox_dir) / f"spiritualist_{key}_predictions.csv")
        results = runner.run_evaluation(model, map_ignore_class=True)
        all_results["models"][name] = results["metrics"]
    with open(output_dir / "benchmark_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(pd.DataFrame(all_results["models"]).T.round(4).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
