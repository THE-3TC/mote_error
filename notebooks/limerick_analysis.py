import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MOTe: a worked example

    The MOTe error is a decomposable evaluation metric for Document Layout Analysis. The MOTe assigns every pixel a single state based on the relationship between prediction and ground truth. There are 6 possible pixel states, which produce the Missing, Overlap and Trespass errors that sum to the MOTe error, plus the support metric Excess.

    - **covered** — ground truth found by exactly one prediction
    - **overlap** — ground truth claimed by more than one
    - **trespass** — ground truth covered by a prediction belonging to another unit
    - **overlap + trespass** — both at once
    - **excess** — background a prediction claimed
    - **missing** — ground truth no prediction found

    The MOTe error builds on the Semantic Structural Unit (SSU) concept, which groups text using the logical groups that come from the narrative flow. As such, the title of a limerick and the limerick itself form a Semantic unit; however, as the title and the text are different classes of text, they each form their own structural unit. As shown in the example below, the first and last limericks are formed of two SSUs each whilst limerick 2 is formed of three as the limerick text is split across both columns.

    This notebook provides a worked visual example of the MOTe error. It uses a series of three limericks on a two-column layout.

    By the end of the notebook, you will have gained an intuitive and practical understanding of the MOTe error and how, using the SSU, it is more expressive than the traditional approach to Document Layout Analysis, the “F1”.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from mote_error import (
        load_limerick_example,
        extract_ssu_boxes,
        extract_line_boxes,
        extract_word_boxes,
        reconstruct_text,
        mote,
        compute_mote_masks,
        visualize_mote_states,
    )
    from mote_error.layout import f1, iou, mean_iou
    from mote_error.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks

    return (
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        compute_mote_masks,
        extract_line_boxes,
        extract_ssu_boxes,
        extract_word_boxes,
        f1,
        load_limerick_example,
        mean_iou,
        mote,
        np,
        patches,
        plt,
        reconstruct_text,
        visualize_mote_states,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving the figures

    Toggle on to write every figure in this notebook to `data/figures/` at
    300 dpi. Off by default, so simply running the notebook writes nothing.
    """)
    return


@app.cell
def _(mo):
    save_figures = mo.ui.checkbox(
        value=False,
        label="Save figures to `data/figures` at 300 dpi",
    )
    save_figures
    return (save_figures,)


@app.cell
def _(save_figures):
    from pathlib import Path

    try:
        _root = Path(__file__).resolve().parent.parent
    except NameError:                       # no __file__ in some marimo contexts
        _root = Path.cwd()
    FIGURE_DIR = _root / "data" / "figures"

    def save_figure(fig, name):
        """Write ``fig`` to data/figures/<name>.png at 300 dpi, if enabled."""
        if not save_figures.value:
            return
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        _path = FIGURE_DIR / f"{name}.png"
        fig.savefig(_path, dpi=300, bbox_inches="tight")
        print(f"saved {_path}")

    return (save_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The ground truth is a character table

    Once loaded the basic image and the groundtruth data table can be inspected. The data table shows a simple strcuture of the bounding box for each letter plus the necessary unique ID's requried to reconstruct the text.
    """)
    return


@app.cell
def _(load_limerick_example, plt, save_figure):
    chars, image, pred_boxes = load_limerick_example()

    _fig, _ax = plt.subplots(figsize=(14, 6))
    _ax.imshow(image, cmap='gray')

    _ax.set_title("Three Limericks")
    _ax.axis('off')
    save_figure(_fig, "basic_limericks")
    plt.show()
    return chars, image, pred_boxes


@app.cell
def _(chars, extract_ssu_boxes, image, pred_boxes):
    gt_boxes = extract_ssu_boxes(chars)

    print(f"{len(chars)} characters   image {image.shape[1]}x{image.shape[0]}   "
          f"{len(pred_boxes)} predictions")
    print()
    print(chars.head(8).to_string(index=False))
    return (gt_boxes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limerick text

    As can be seen below the text can be reconstructed to produce raw text of the three limericks
    """)
    return


@app.cell
def _(chars, reconstruct_text):
    """The page text, rebuilt from character positions and word groupings alone."""
    print(reconstruct_text(chars))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualising the underlying Semantic Units

    Although simple it can help to show the limerick image as a coloured by Semantic Unit. This visualisation clearly shows that whilst the second limerick is split across three spatially separate boxes it is represented as a single semantic whole,
    """)
    return


@app.cell
def _(chars, gt_boxes, image, np, patches, plt, save_figure):
    """Ground truth coloured by semantic unit: the split poem shares a colour."""


    _fig, _ax = plt.subplots(figsize=(14, 6))
    _ax.imshow(image, cmap='gray')

    _sems = sorted(chars.semantic_unit.unique())
    _colours = plt.cm.Set2(np.linspace(0, 1, max(len(_sems), 3)))

    for _b in gt_boxes:
        _c = _colours[_sems.index(_b['semantic_unit'])]
        _ax.add_patch(patches.Rectangle(
            (_b['x'], _b['y']), _b['width'], _b['height'],
            linewidth=2, edgecolor=_c, facecolor=_c, alpha=0.30,
        ))
        _ax.text(_b['x'] + 4, _b['y'] + 14, f"ssu {_b['ssu_id']}",
                 fontsize=9, color='black')

    _ax.set_title("Ground truth SSUs, coloured by semantic unit")
    _ax.axis('off')
    save_figure(_fig, "ssu_semantic_units")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Granularity

    A major issue with using the F1 as a quality metric for text is that it is very sensitive to the granularity of the predictions and the ground truth. Text can be show at different granularities with no single method being necessarily better than the other.

    The box below shows the bounding boxes at three levels of granularity. In some cases bounding boxes can be at character level.
    """)
    return


@app.cell
def _(chars, extract_line_boxes, extract_ssu_boxes, extract_word_boxes):
    line_boxes = extract_line_boxes(chars)
    word_boxes = extract_word_boxes(chars)

    print(f"  SSUs  {len(extract_ssu_boxes(chars)):3d}")
    print(f"  lines {len(line_boxes):3d}")
    print(f"  words {len(word_boxes):3d}")
    return line_boxes, word_boxes


@app.cell
def _(gt_boxes, image, line_boxes, patches, plt, save_figure, word_boxes):
    _fig, _axes = plt.subplots(3, 1, figsize=(14, 16))

    for _ax, _boxes, _title, _colour in [
        (_axes[0], gt_boxes,   "SSU",  'tab:blue'),
        (_axes[1], line_boxes, "line", 'tab:orange'),
        (_axes[2], word_boxes, "word", 'tab:green'),
    ]:
        _ax.imshow(image, cmap='gray')
        for _b in _boxes:
            _ax.add_patch(patches.Rectangle(
                (_b['x'], _b['y']), _b['width'], _b['height'],
                linewidth=1.2, edgecolor=_colour, facecolor='none',
            ))
        _ax.set_title(f"{_title} level — {len(_boxes)} regions")
        _ax.axis('off')

    plt.tight_layout()
    save_figure(_fig, "granularity_levels")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pragmatic Error and Pragmatic Competence: How F1 and MOTe respond to granularity mismatches

    IoU based prediction metrics such as the F1 match predictions to ground truth on a one-to-one basis using an IoU
    threshold. If the prediction and ground truth have significantly different levels of granularity, then the IoU becomes small, and the score collapses, even if the overall prediction is very high. This failure of the evaluation metric to understand the model/data relationship is a ‘**Pragmatic failure**’.

    In contrast, the MOTe error has a many-to-one relationship: many predictions can map to a single SSU. This makes the MOTe error substantially more robust to granularity differences,, giving it higher ‘**Pragmatic competence**’, or ability to interpret the model/data relationship, than its IoU-based counterparts.

    Below, the line-level ground truth is used as the prediction for the SSU-level ground truth, and vice versa. This means that both variants should get perfect scores.
    However, the mean IoU and the F1 score are very bad for both. In contrast, the MOTe error is zero when lines are used as GT, and very low when lines are used as predictions. This difference is because the line predictions miss empty whitespace created by SSU that creates boxes based on the maximum extent of the characters they contain.
    Despite this small error, the MOTe error gives a substantially more granularity robust result than the F1.
    """)
    return


@app.cell
def _(
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    f1,
    gt_boxes,
    image,
    line_boxes,
    mean_iou,
    mote,
):
    _H, _W = image.shape[:2]

    def _score(gt, preds, label):
    # Print out the F1, mean IoU and MOTe error given prediction and Ground Truth masks.
        _map = boxes_to_gt_ssu_map(gt, _W, _H, _W, _H)
        _masks = boxes_to_pred_masks(preds, _W, _H, _W, _H)
        _mote, _M, _O, _T, _E = mote(_map, _masks)
        print(f"  {label}")
        print(f"    F1@0.5   {f1(preds, gt):.4f}     mIoU {mean_iou(preds, gt):.4f}")
        print(f"    MOTe     {_mote:.4f}     missing {_M:.4f}  overlap {_O:.4f}  "
              f"trespass {_T:.4f}  excess {_E:.4f}")

    _score(gt_boxes, line_boxes, "GT = SSU, predictions = lines")
    print()
    _score(line_boxes, gt_boxes, "GT = lines, predictions = SSUs")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predicting with MOTe

    Using the example predictions loaded at the beggining of the notebook we overlay them on top of the SSU ground truth and visualise the resulting pixel classifications
    """)
    return


@app.cell
def _(
    compute_mote_masks,
    gt_ssu_map,
    image,
    plt,
    pred_masks,
    save_figure,
    visualize_mote_states,
):
    mote_masks = compute_mote_masks(gt_ssu_map, pred_masks)

    _fig, _ax = plt.subplots(figsize=(14, 6))
    _ax.imshow(image, cmap='gray')
    _patches = visualize_mote_states(image, mote_masks, ax=_ax)
    _ax.legend(handles=_patches, loc='lower center', fontsize=9, ncol = 6)
    _ax.set_title("MOTe pixel states")
    _ax.axis('off')
    save_figure(_fig, "mote_pixel_states")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Quantifying the results

    Whilst the visualisation provides clear insight into the prediction, the quantification below helps us understand what the prediction means.

    By decomposing the overall MOTe error of 0.33 into its constituent parts, we can start seeing where the model's strengths and weaknesses lie. In this case, although Missing is low at 0.09, there is significant Overlap (0.12) and Trespass (0.12). This way, we are able to start thinking about whether the errors produced at the Document Layout Analysis stage will have a negative impact downstream for the task we are trying to perform.
    """)
    return


@app.cell
def _(
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    f1,
    gt_boxes,
    image,
    mean_iou,
    mote,
    pred_boxes,
):
    IMG_H, IMG_W = image.shape[:2]

    gt_ssu_map = boxes_to_gt_ssu_map(gt_boxes, IMG_W, IMG_H, IMG_W, IMG_H)
    pred_masks = boxes_to_pred_masks(pred_boxes, IMG_W, IMG_H, IMG_W, IMG_H)

    error, missing, overlap, trespass, excess = mote(gt_ssu_map, pred_masks)

    print(f"  MOTe       {error:.4f}")
    print(f"    missing  {missing:.4f}   of the ground truth was not found")
    print(f"    overlap  {overlap:.4f}   was claimed by more than one prediction")
    print(f"    trespass {trespass:.4f}   was attributed to the wrong unit")
    print(f"    excess   {excess:.4f}   of the background was claimed")
    print()
    print(f"  F1@0.5     {f1(pred_boxes, gt_boxes):.4f}")
    print(f"  mIoU       {mean_iou(pred_boxes, gt_boxes):.4f}")
    return gt_ssu_map, pred_masks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Character positions

    The table also carries every character's position, which is what the spatial OCR
    decompositions consume. `chars_to_region_chars(chars)` returns the
    `RegionChars` that `cdd_decomp_spatial` expects, so text-level error can be
    attributed to the same regions the layout score was computed over.
    """)
    return


@app.cell
def _(chars):
    from mote_error import chars_to_region_chars

    region_chars = chars_to_region_chars(chars)

    print(f"  {len(region_chars.tokens)} characters")
    print(f"  regions: {sorted(set(region_chars.region_ids.tolist()))}")
    print(f"  x range: {region_chars.xs.min()}–{region_chars.xs.max()}")
    print(f"  y range: {region_chars.ys.min()}–{region_chars.ys.max()}")
    return


if __name__ == "__main__":
    app.run()
