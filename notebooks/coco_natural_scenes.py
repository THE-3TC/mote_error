import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MOTe on natural scenes

    Although MOTe was developed for document understanding pipelines, there is nothing that prevents it being used in general machine vision tasks. In this notebook we compare the performance of MOTe to Panoptic Quality (PQ), on two images from COCO panoptic. We apply two models YOLO26-instance and Mask2Former panoptic. In Panoptic segmentation each pixel can only have a single class, meaning that Overlap will become zero, in instance segmentation, Overlap is likely low but will still be present.

    **Panoptic Quality (PQ)**, the standard metric for this task.

    PQ matches each predicted segment to at most one ground-truth segment of the same class at
    IoU > 0.5. Its components:

    - **SQ** (segmentation quality): mean IoU of the matched pairs.
    - **RQ** (recognition quality): an F1 score over segments.

    $$\text{PQ} = \text{SQ} \times \text{RQ}$$

    PQ is an instance-matching metric, so it inherits F1's sensitivity to granularity. It also
    requires non-overlapping predictions, so it cannot say anything about overlap. MOTe is
    computed over pixels and has neither restriction.


    This notebook is a simple demonstration of MOTe on natural scenes deeper analysis is beyond it's scope/
    """)
    return


@app.cell
def _():
    # --- Point these at your data -------------------------------------------------
    DATA_DIR = "data/coco_data"  # <name>_<image_id>.jpg and <image_id>.png panoptic masks
    GT_JSON = "data/coco_data/coco_panoptic_gt.json"  # scripts/extract_coco_panoptic_gt.py
    PRED_DIR = "data/coco_data"  # <model>/<image_id>/ from scripts/run_coco_panoptic_models.py

    # Segments to split into separate SSUs for the "split" GT variant: {image_id: category}.
    # COCO labels the two trains in 233341 as one instance; whether they are one coupled
    # train or two parked trains is not clear from the image.
    SPLIT_SEGMENTS = {233341: "train"}

    # Display names for tables and figures
    IMAGE_NAMES = {325871: "Elephant", 233341: "Train"}
    MODEL_NAMES = {"yolo26": "YOLO26", "mask2former": "Mask2Former"}

    # Scope used for the LaTeX tables and saved figures; "things" is fair to both models
    PAPER_SCOPE = "things"
    return (
        DATA_DIR,
        GT_JSON,
        IMAGE_NAMES,
        MODEL_NAMES,
        PAPER_SCOPE,
        PRED_DIR,
        SPLIT_SEGMENTS,
    )


@app.cell
def _():
    import json
    from collections import defaultdict
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    from pymote.class_metrics import class_confusion_counts, finalize_class_counts
    from pymote.layout import mote
    from pymote.types import MaskInstance
    from pymote.visualisation import compute_mote_masks, visualize_mote_states
    return (
        Image,
        MaskInstance,
        Path,
        class_confusion_counts,
        compute_mote_masks,
        mote,
        defaultdict,
        finalize_class_counts,
        json,
        np,
        pd,
        plt,
        visualize_mote_states,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Ground truth

    COCO panoptic stores each image's segmentation as an RGB PNG. The segment id is
    `R + 256·G + 256²·B`, and id `0` is **void** (unlabelled). The JSON's `segments_info`
    maps each segment id to a category, marks it as a *thing* (countable: person, elephant)
    or *stuff* (amorphous: grass, sky), and flags `iscrowd` regions.
    """)
    return


@app.cell
def _(DATA_DIR, GT_JSON, IMAGE_NAMES, Image, Path, json, np):
    def rgb2id(png):
        png = png.astype(np.uint32)
        return png[..., 0] + 256 * png[..., 1] + 256 * 256 * png[..., 2]

    with open(GT_JSON) as _f:
        _gt = json.load(_f)

    categories = {c["id"]: c for c in _gt["categories"]}
    name_to_cat = {c["name"]: c["id"] for c in _gt["categories"]}

    samples = {}
    for _ann in _gt["annotations"]:
        _id = _ann["image_id"]
        _jpg = next(Path(DATA_DIR).glob(f"*_{_id}.jpg"))
        _png = np.array(Image.open(Path(DATA_DIR) / _ann["file_name"]).convert("RGB"))
        samples[_id] = {
            "image_id": _id,
            "name": IMAGE_NAMES.get(_id, str(_id)),
            "image_path": _jpg,
            "image": np.array(Image.open(_jpg).convert("RGB")),
            "pan_map": rgb2id(_png),
            "segments": _ann["segments_info"],
        }
    print(f"Loaded {len(samples)} images: {sorted(samples)}")
    return categories, name_to_cat, samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Alignment check

    The mask and the image must have the same shape. Every id in the PNG must also appear in
    `segments_info`, and each segment's pixel count must match its recorded `area`. A failure
    here means the wrong PNG or a rescaled image.
    """)
    return


@app.cell
def _(np, pd, samples):
    _rows = []
    for _id, _s in samples.items():
        _ids_in_png = set(np.unique(_s["pan_map"]).tolist()) - {0}
        _ids_in_json = {seg["id"] for seg in _s["segments"]}
        _area_err = max(
            abs(int((_s["pan_map"] == seg["id"]).sum()) - seg["area"]) for seg in _s["segments"]
        )
        _rows.append(
            {
                "image_id": _id,
                "image_hw": _s["image"].shape[:2],
                "mask_hw": _s["pan_map"].shape,
                "shape_match": _s["image"].shape[:2] == _s["pan_map"].shape,
                "n_segments": len(_s["segments"]),
                "ids_match": _ids_in_png == _ids_in_json,
                "max_area_error_px": _area_err,
                "void_fraction": round(float((_s["pan_map"] == 0).mean()), 3),
            }
        )
    alignment_df = pd.DataFrame(_rows)
    assert alignment_df.shape_match.all(), "Image and panoptic mask shapes differ"
    assert alignment_df.ids_match.all(), "Segment ids in PNG and JSON differ"
    alignment_df
    return


@app.cell
def _(categories, np):
    GREY = (128, 128, 128)

    def cat_colour(cat_id):
        # The release's categories carry no colours, so seed one from the id: stable
        # across images and models.
        if cat_id not in categories:
            return GREY
        return tuple(np.random.default_rng(cat_id).integers(40, 255, size=3))

    def overlay_masks(image, masks, colours, alpha=0.6):
        """Blend coloured masks over an image and draw white segment boundaries."""
        out = image.astype(float).copy()
        label = np.zeros(image.shape[:2], dtype=np.int32)
        for _i, (_m, _c) in enumerate(zip(masks, colours), start=1):
            out[_m] = (1 - alpha) * out[_m] + alpha * np.asarray(_c, dtype=float)
            label[_m] = _i
        edges = np.zeros(label.shape, dtype=bool)
        edges[:-1] |= label[:-1] != label[1:]
        edges[:, :-1] |= label[:, :-1] != label[:, 1:]
        out[edges] = 255
        return out.astype(np.uint8)

    def label_segments(ax, masks, names, min_frac=0.005):
        """Write each segment's name on the mask pixel nearest its centroid."""
        for _m, _name in zip(masks, names):
            if _m.mean() < min_frac:
                continue
            _ys, _xs = np.nonzero(_m)
            _k = np.argmin((_ys - _ys.mean()) ** 2 + (_xs - _xs.mean()) ** 2)
            ax.text(
                _xs[_k],
                _ys[_k],
                _name,
                fontsize=8,
                ha="center",
                va="center",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none", "pad": 1},
            )

    def cat_name(cat_id):
        return categories[cat_id]["name"] if cat_id in categories else "unmapped"
    return cat_colour, cat_name, label_segments, overlay_masks


@app.cell
def _(cat_colour, cat_name, label_segments, overlay_masks, plt, samples):
    _fig, _axes = plt.subplots(len(samples), 2, figsize=(12, 7 * len(samples)), squeeze=False)
    for (_id, _s), (_ax_img, _ax_gt) in zip(samples.items(), _axes):
        _masks = [_s["pan_map"] == seg["id"] for seg in _s["segments"]]
        _cats = [seg["category_id"] for seg in _s["segments"]]
        _ax_img.imshow(_s["image"])
        _ax_img.set_title(f"{_s['image_path'].name}  {_s['image'].shape[1]}×{_s['image'].shape[0]}")
        _ax_gt.imshow(overlay_masks(_s["image"], _masks, [cat_colour(c) for c in _cats]))
        label_segments(_ax_gt, _masks, [cat_name(c) for c in _cats])
        _ax_gt.set_title(f"Panoptic GT: {len(_masks)} segments (void left unpainted)")
        for _ax in (_ax_img, _ax_gt):
            _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One train or two?

    COCO labels the two trains in image 233341 as a **single** `train` instance. From the
    image it is not clear whether this is one coupled train or two trains parked nose to
    nose. What counts as one unit is a labelling decision, and it is exactly what SSUs make
    explicit.

    To see how much each metric depends on that decision, everything below is scored against
    two versions of the ground truth:

    - `coco`: the annotation as published.
    - `split`: each segment named in `SPLIT_SEGMENTS` is cut at the widest run of empty
      columns inside it, giving one SSU per train.

    The annotation also includes the "Solingen Hbf" station sign in the right-hand train. Both
    variants keep it, so the models are penalised slightly for correctly leaving it out.
    """)
    return


@app.cell
def _(SPLIT_SEGMENTS, categories, np):
    GT_VARIANTS = ("coco", "split")

    def split_at_gap(mask):
        """Split a mask into left and right parts at its widest run of empty columns."""
        cols = np.nonzero(mask.any(axis=0))[0]
        gaps = np.diff(cols)
        k = int(np.argmax(gaps))
        if gaps[k] <= 1:
            raise ValueError("Segment has no empty-column gap to split at")
        left = mask.copy()
        left[:, (cols[k] + cols[k + 1]) // 2 :] = False
        return [left, mask & ~left]

    def gt_segments(sample, variant="coco"):
        split_name = SPLIT_SEGMENTS.get(sample["image_id"]) if variant == "split" else None
        segs = []
        for seg in sample["segments"]:
            base = {
                "mask": sample["pan_map"] == seg["id"],
                "category_id": seg["category_id"],
                "isthing": bool(categories[seg["category_id"]]["isthing"]),
                "iscrowd": bool(seg["iscrowd"]),
            }
            if categories[seg["category_id"]]["name"] == split_name:
                segs += [dict(base, mask=_m) for _m in split_at_gap(base["mask"])]
            else:
                segs.append(base)
        return segs
    return GT_VARIANTS, gt_segments


@app.cell
def _(
    GT_VARIANTS,
    SPLIT_SEGMENTS,
    cat_name,
    gt_segments,
    label_segments,
    overlay_masks,
    plt,
    samples,
):
    _fig, _axes = plt.subplots(
        len(SPLIT_SEGMENTS), len(GT_VARIANTS), figsize=(12, 6 * len(SPLIT_SEGMENTS)), squeeze=False
    )
    for _id, _row in zip(SPLIT_SEGMENTS, _axes):
        _s = samples[_id]
        for _variant, _ax in zip(GT_VARIANTS, _row):
            _things = [g for g in gt_segments(_s, _variant) if g["isthing"]]
            _masks = [g["mask"] for g in _things]
            # Colour by instance, not category, so the split is visible.
            _cmap = plt.get_cmap("tab10")
            _colours = [tuple(int(255 * v) for v in _cmap(_i)[:3]) for _i in range(len(_masks))]
            _ax.imshow(overlay_masks(_s["image"], _masks, _colours))
            label_segments(_ax, _masks, [cat_name(g["category_id"]) for g in _things])
            _ax.set_title(f"{_id}, GT = {_variant}: {len(_things)} thing SSUs")
            _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Predictions

    Every prediction is a dict `{"mask", "category_id", "score"}` at the image's native
    resolution. Real model outputs are loaded from `PRED_DIR/<model>/<image_id>` (an `.npz`, or a folder of `.npy` files). Class
    names are mapped to COCO panoptic category ids by name, and anything unmapped gets `-1`.
    """)
    return


@app.cell
def _(PRED_DIR, Path, name_to_cat, np):
    predictions = {}  # model -> image_id -> list of prediction dicts

    def _load_arrays(path):
        """A prediction is either <image_id>.npz or an unzipped <image_id>/ of .npy files."""
        if path.suffix == ".npz":
            return np.load(path)
        return {_k: np.load(path / f"{_k}.npy") for _k in ("masks", "labels", "scores")}

    _unmapped = set()
    _root = Path(PRED_DIR)
    _model_dirs = [p for p in sorted(_root.iterdir()) if p.is_dir()] if _root.exists() else []
    for _model_dir in _model_dirs:
        for _f in sorted(_model_dir.iterdir()):
            if not _f.stem.isdigit():
                continue
            _d = _load_arrays(_f)
            _preds = []
            for _m, _label, _score in zip(_d["masks"], _d["labels"], _d["scores"]):
                _cat = name_to_cat.get(str(_label), -1)
                if _cat == -1:
                    _unmapped.add(str(_label))
                _preds.append({"mask": _m, "category_id": _cat, "score": float(_score)})
            predictions.setdefault(_model_dir.name, {})[int(_f.stem)] = _preds

    if _unmapped:
        print(f"WARNING: labels with no COCO panoptic category: {sorted(_unmapped)}")
    print("Models:", ", ".join(predictions))
    return (predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Scoring

    Both metrics are computed at two scopes:

    - `all`: things and stuff. This is the natural scope for a panoptic model.
    - `things`: stuff ground truth becomes background and stuff predictions are dropped.
      This is the fair scope for an instance model like YOLO, which never predicts stuff.

    At the `all` scope the only background is the void, a small fraction of the image. Excess
    is normalised by background area, so it swings sharply there.
    """)
    return


@app.cell
def _(defaultdict, np):
    def panoptic_quality(gt_segs, pred_segs, void):
        """COCO panoptic quality for one image. Returns (PQ, SQ, RQ)."""
        taken = np.zeros(void.shape, dtype=bool)
        preds = []
        for p in sorted(pred_segs, key=lambda p: -p["score"]):
            m = p["mask"] & ~taken
            if m.sum() == 0 or m.sum() < 0.5 * p["mask"].sum():
                continue
            taken |= m
            preds.append(dict(p, mask=m))

        stats = defaultdict(lambda: {"iou": 0.0, "tp": 0, "fp": 0, "fn": 0})
        crowd = defaultdict(lambda: np.zeros(void.shape, dtype=bool))
        matched = set()
        for g in gt_segs:
            if g["iscrowd"]:
                crowd[g["category_id"]] |= g["mask"]
                continue
            for pi, p in enumerate(preds):
                if pi in matched or p["category_id"] != g["category_id"]:
                    continue
                inter = (g["mask"] & p["mask"]).sum()
                union = g["mask"].sum() + p["mask"].sum() - inter - (p["mask"] & void).sum()
                iou = inter / union if union else 0.0
                if iou > 0.5:
                    stats[g["category_id"]]["tp"] += 1
                    stats[g["category_id"]]["iou"] += iou
                    matched.add(pi)
                    break
            else:
                stats[g["category_id"]]["fn"] += 1

        for pi, p in enumerate(preds):
            if pi in matched:
                continue
            ignored = (p["mask"] & void).sum() + (p["mask"] & crowd[p["category_id"]]).sum()
            if ignored / p["mask"].sum() > 0.5:
                continue
            stats[p["category_id"]]["fp"] += 1

        pq, sq, rq = [], [], []
        for st in stats.values():
            denom = st["tp"] + 0.5 * st["fp"] + 0.5 * st["fn"]
            if denom == 0:
                continue
            pq.append(st["iou"] / denom)
            sq.append(st["iou"] / st["tp"] if st["tp"] else 0.0)
            rq.append(st["tp"] / denom)
        if not pq:
            return 1.0, 1.0, 1.0
        return float(np.mean(pq)), float(np.mean(sq)), float(np.mean(rq))
    return (panoptic_quality,)


@app.cell
def _(
    GT_VARIANTS,
    MaskInstance,
    categories,
    class_confusion_counts,
    mote,
    finalize_class_counts,
    gt_segments,
    np,
    panoptic_quality,
    pd,
    predictions,
    samples,
):
    SCOPES = ("all", "things")

    def mote_f1(gt_map, gt, preds):
        """Class-aware MOTe F1: micro F1 of per-class pixel coverage precision and recall.

        The class list is the union of GT and predicted classes, so pixels predicted as a
        class absent from the GT count against precision rather than being dropped.
        """
        ssu_to_class = {_ssu: g["category_id"] for _ssu, g in enumerate(gt, start=1)}
        masks = [
            MaskInstance(mask=p["mask"], label=p["category_id"], pred_id=i)
            for i, p in enumerate(preds)
        ]
        classes = sorted(set(ssu_to_class.values()) | {m.label for m in masks})
        counts = class_confusion_counts(gt_map, ssu_to_class, masks, classes)
        return finalize_class_counts(counts).micro_f1

    def in_scope(cat_id, scope):
        if scope == "all":
            return True
        return cat_id in categories and bool(categories[cat_id]["isthing"])

    def scoped_inputs(sample, preds, scope, variant):
        """GT SSU map, GT segments and predictions restricted to one scope."""
        gt = [g for g in gt_segments(sample, variant) if in_scope(g["category_id"], scope)]
        gt_map = np.zeros(sample["pan_map"].shape, dtype=np.int32)
        for _ssu, g in enumerate(gt, start=1):
            gt_map[g["mask"]] = _ssu
        return gt_map, gt, [p for p in preds if in_scope(p["category_id"], scope)]

    _rows = []
    for _model, _by_image in predictions.items():
        for _id, _preds in _by_image.items():
            _s = samples[_id]
            _void = _s["pan_map"] == 0
            for _variant in GT_VARIANTS:
                for _scope in SCOPES:
                    _gt_map, _gt, _p = scoped_inputs(_s, _preds, _scope, _variant)
                    _masks = [MaskInstance(mask=p["mask"], pred_id=i) for i, p in enumerate(_p)]
                    if not _masks:
                        _masks = [MaskInstance(mask=np.zeros_like(_void), pred_id=0)]
                    _mote, _m, _o, _t, _e = mote(_gt_map, _masks)
                    _pq, _sq, _rq = panoptic_quality(_gt, _p, _void)
                    _rows.append(
                        {
                            "model": _model,
                            "image": _s["name"],
                            "image_id": _id,
                            "gt_variant": _variant,
                            "scope": _scope,
                            "mote": _mote,
                            "missing": _m,
                            "overlap": _o,
                            "trespass": _t,
                            "excess": _e,
                            "mote_f1": mote_f1(_gt_map, _gt, _p),
                            "pq": _pq,
                            "sq": _sq,
                            "rq": _rq,
                            "n_gt": len(_gt),
                            "n_pred": len(_p),
                        }
                    )
    scores_df = pd.DataFrame(_rows)
    scores_df.drop(columns="image_id").round(3)
    return SCOPES, scoped_inputs, scores_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Does the answer depend on the train question?

    The same predictions are scored against both ground truths at the `things` scope, which
    is fair to both models.

    - **Mask2Former** predicts one mask per train. Each mask sits inside a single SSU under
      either ground truth, so MOTe does not change: predicting finer units than the ground
      truth costs nothing. PQ does change, because the one-to-one match fails against the
      single COCO segment.
    - **YOLO26** predicts one mask per train plus a low-confidence mask covering both. That
      mask is overlap under either ground truth. Under `split` it also crosses from one train
      into the other, so MOTe adds trespass. PQ never sees this mask: making the predictions
      non-overlapping removes it before matching, so PQ *rises* under `split`.
    """)
    return


@app.cell
def _(SPLIT_SEGMENTS, scores_df):
    _cols = [
        "mote", "missing", "overlap", "trespass", "mote_f1", "pq", "sq", "rq", "n_gt", "n_pred"
    ]
    variant_df = (
        scores_df[scores_df.image_id.isin(SPLIT_SEGMENTS) & (scores_df.scope == "things")]
        .set_index(["image", "model", "gt_variant"])[_cols]
        .sort_index()
        .round(3)
    )
    variant_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MOTe against PQ

    One point per image. As PQ is a score and MOTe an error, the dashed line marks agreement
    (MOTe = 1 − PQ). Points below the dashed line are cases where MOTe rates the prediction
    more favourably than PQ, typically a granularity disagreement. Points above it are cases
    where MOTe is penalising overlap or trespass that PQ cannot see.
    """)
    return


@app.cell
def _(GT_VARIANTS, SCOPES, mo):
    scope_picker = mo.ui.dropdown(options=list(SCOPES), value="all", label="Scope")
    gt_picker = mo.ui.dropdown(options=list(GT_VARIANTS), value="coco", label="GT")
    mo.hstack([scope_picker, gt_picker], justify="start")
    return gt_picker, scope_picker


@app.cell
def _(gt_picker, plt, scope_picker, scores_df):
    _df = scores_df[(scores_df.scope == scope_picker.value) & (scores_df.gt_variant == gt_picker.value)]
    _models = list(dict.fromkeys(scores_df.model))  # fixed order, so colours never shift
    _markers = dict(zip(list(dict.fromkeys(scores_df.image)), ["o", "s", "^", "D"]))
    _first = next(iter(_markers))
    _cmap = plt.get_cmap("tab10")

    _fig, _ax = plt.subplots(figsize=(7, 7))
    for _i, _model in enumerate(_models):
        for _image, _grp in _df[_df.model == _model].groupby("image"):
            _ax.scatter(
                _grp.pq,
                _grp.mote,
                color=_cmap(_i % 10),
                marker=_markers[_image],
                s=70,
                edgecolor="white",
                linewidth=1,
                label=_model if _image == _first else None,
            )
    for _image, _mk in _markers.items():
        _ax.scatter([], [], marker=_mk, color="grey", label=_image)
    #_ax.plot([0, 1], [1, 0], "k--", linewidth=1)
    #_ax.set_xlim(-0.05, 1.05)
    _ax.set_xlabel("PQ")
    _ax.set_ylabel("MOTe error")
    _ax.set_title(f"MOTe vs PQ, scope = {scope_picker.value}, GT = {gt_picker.value}")
    _ax.grid(alpha=0.3)
    _ax.legend(fontsize=8, loc="lower right")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Where the pixels went

    Pick a model, an image and a scope. The left panel shows the ground truth at that scope,
    the middle panel shows the predictions coloured by class, and the right panel shows the
    MOTe pixel states:

    | colour | state |
    |---|---|
    | green | covered |
    | amber | overlap |
    | red | trespass |
    | purple | overlap + trespass |
    | grey | missing |
    | blue | excess |
    """)
    return


@app.cell
def _(
    MaskInstance,
    cat_colour,
    cat_name,
    compute_mote_masks,
    label_segments,
    np,
    overlay_masks,
    visualize_mote_states,
):
    def draw_segments(ax, sample, segs):
        """Segments over the image, coloured and labelled by category."""
        masks = [g["mask"] for g in segs]
        ax.imshow(overlay_masks(sample["image"], masks, [cat_colour(g["category_id"]) for g in segs]))
        label_segments(ax, masks, [cat_name(g["category_id"]) for g in segs])
        ax.axis("off")

    def draw_mote(ax, sample, gt_map, preds):
        """MOTe pixel states over the image, with a legend."""
        masks = [MaskInstance(mask=p["mask"], pred_id=i) for i, p in enumerate(preds)] or [
            MaskInstance(mask=np.zeros(gt_map.shape, dtype=bool), pred_id=0)
        ]
        patches = visualize_mote_states(
            sample["image"], compute_mote_masks(gt_map, masks), ax=ax, show_missing=True
        )
        ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.9)
        ax.axis("off")
    return draw_mote, draw_segments


@app.cell
def _(GT_VARIANTS, MODEL_NAMES, SCOPES, mo, predictions, samples):
    model_picker = mo.ui.dropdown(
        options={MODEL_NAMES.get(m, m): m for m in predictions},
        value=MODEL_NAMES.get(next(iter(predictions)), next(iter(predictions))),
        label="Model",
    )
    image_picker = mo.ui.dropdown(
        options={s["name"]: i for i, s in samples.items()},
        value=next(iter(samples.values()))["name"],
        label="Image",
    )
    view_scope_picker = mo.ui.dropdown(options=list(SCOPES), value="all", label="Scope")
    view_gt_picker = mo.ui.dropdown(options=list(GT_VARIANTS), value="coco", label="GT")
    mo.hstack([model_picker, image_picker, view_scope_picker, view_gt_picker], justify="start")
    return image_picker, model_picker, view_gt_picker, view_scope_picker


@app.cell
def _(
    draw_mote,
    draw_segments,
    image_picker,
    model_picker,
    plt,
    predictions,
    samples,
    scoped_inputs,
    scores_df,
    view_gt_picker,
    view_scope_picker,
):
    _id = image_picker.value
    _model = model_picker.value
    _scope = view_scope_picker.value
    _variant = view_gt_picker.value
    _s = samples[_id]
    _gt_map, _gt, _p = scoped_inputs(_s, predictions[_model].get(_id, []), _scope, _variant)
    _row = scores_df[
        (scores_df.model == _model)
        & (scores_df.image_id == _id)
        & (scores_df.scope == _scope)
        & (scores_df.gt_variant == _variant)
    ].iloc[0]

    _fig, (_ax_gt, _ax_pred, _ax_mote) = plt.subplots(1, 3, figsize=(20, 9))
    draw_segments(_ax_gt, _s, _gt)
    _ax_gt.set_title(f"GT ({_variant}, {_scope}): {len(_gt)} segments")
    draw_segments(_ax_pred, _s, _p)
    _ax_pred.set_title(f"{_model}: {len(_p)} segments")
    draw_mote(_ax_mote, _s, _gt_map, _p)
    _ax_mote.set_title(
        f"MOTe {_row.mote:.3f} (M {_row.missing:.2f}, O {_row.overlap:.2f}, "
        f"T {_row.trespass:.2f}, E {_row.excess:.2f})\n"
        f"PQ {_row.pq:.3f} (SQ {_row.sq:.2f}, RQ {_row.rq:.2f})"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Tables and figures for the paper

    One LaTeX table per image, and one untitled figure each for the ground truth, each
    model's predictions and each model's MOTe pixel states. Both use the published COCO ground truth
    (not the `split` version) at `PAPER_SCOPE`.

    Toggle on to write them to `data/figures/coco_natural_scenes/`, figures at 300 dpi. Off
    by default, so simply running the notebook writes nothing.
    """)
    return


@app.cell
def _(mo):
    save_outputs = mo.ui.checkbox(
        value=False,
        label="Save tables and figures to `data/figures/coco_natural_scenes`",
    )
    save_outputs
    return (save_outputs,)


@app.cell
def _(Path, save_outputs):
    try:
        _root = Path(__file__).resolve().parent.parent
    except NameError:  # no __file__ in some marimo contexts
        _root = Path.cwd()
    OUTPUT_DIR = _root / "data" / "figures" / "coco_natural_scenes"

    def save_figure(fig, name):
        """Write ``fig`` to OUTPUT_DIR/<name>.png at 300 dpi, if enabled."""
        if not save_outputs.value:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _path = OUTPUT_DIR / f"{name}.png"
        fig.savefig(_path, dpi=300, bbox_inches="tight")
        print(f"saved {_path}")

    def save_text(text, name):
        """Write ``text`` to OUTPUT_DIR/<name>, if enabled."""
        if not save_outputs.value:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / name).write_text(text)
        print(f"saved {OUTPUT_DIR / name}")
    return save_figure, save_text


@app.cell
def _(PAPER_SCOPE, scores_df):
    scores_df[(scores_df.gt_variant == "coco") & (scores_df.scope == PAPER_SCOPE)]
    return


@app.cell
def _(MODEL_NAMES, PAPER_SCOPE, mo, save_text, scores_df):
    import latex_tables as _lt

    _cols = {
        "mote": "MOTe",
        "missing": "Missing",
        "overlap": "Overlap",
        "trespass": "Trespass",
        "excess": "Excess",
        "mote_f1": "MOTe F1",
        "pq": "PQ",
        "sq": "SQ",
        "rq": "RQ",
    }
    _df = scores_df[(scores_df.gt_variant == "coco") & (scores_df.scope == PAPER_SCOPE)]

    _tex = []
    for _image, _grp in _df.groupby("image", sort=False):
        _body = _lt.bold_best_cols(
            _grp.set_index(_grp.model.map(lambda m: MODEL_NAMES.get(m, m)))[list(_cols)]
            .astype(float)
            .rename(columns=_cols)
            .rename_axis("Model"),
            lower_cols=["MOTe", "Missing", "Overlap", "Trespass", "Excess"],
            fmt="{:.2f}",
        )
        _table = _lt.latex_table(
            _body,
            caption=(
                f"MOTe and panoptic quality on the COCO {_image.lower()} image "
                f"({PAPER_SCOPE} scope). MOTe, missing, overlap, trespass and excess are lower-is-better."
            ),
            label=f"tab:coco_{_image.lower()}",
            echo=False,
        )
        save_text(_table, f"table_{_image.lower()}.tex")
        _tex.append(_table)

    mo.md("\n\n".join(f"```latex\n{t}\n```" for t in _tex))
    return


@app.cell
def _(
    MODEL_NAMES,
    PAPER_SCOPE,
    draw_mote,
    draw_segments,
    mo,
    plt,
    predictions,
    samples,
    save_figure,
    scoped_inputs,
):
    def _single(sample):
        """A figure with one axes sized to the image's aspect ratio."""
        _h, _w = sample["image"].shape[:2]
        return plt.subplots(figsize=(6, 6 * _h / _w))

    _rows = []
    for _id, _s in samples.items():
        _stem = _s["name"].lower()
        _row = []

        _gt_map, _gt, _ = scoped_inputs(_s, [], PAPER_SCOPE, "coco")
        _fig, _ax = _single(_s)
        draw_segments(_ax, _s, _gt)
        save_figure(_fig, f"gt_{_stem}")
        _row.append(_fig)

        for _model, _by_image in predictions.items():
            _name = MODEL_NAMES.get(_model, _model)
            _, _, _p = scoped_inputs(_s, _by_image.get(_id, []), PAPER_SCOPE, "coco")

            _fig, _ax = _single(_s)
            draw_segments(_ax, _s, _p)
            save_figure(_fig, f"pred_{_stem}_{_model}")
            _row.append(_fig)

            _fig, _ax = _single(_s)
            draw_mote(_ax, _s, _gt_map, _p)
            save_figure(_fig, f"mote_{_stem}_{_model}")
            _row.append(_fig)

        _rows.append(mo.hstack(_row, justify="start"))
    mo.vstack(_rows)
    return


if __name__ == "__main__":
    app.run()
