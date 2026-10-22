import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import json
    import marimo as mo
    #import patchworklib as pw
    from plotnine import (
        aes,
        geom_hline,
        geom_line,
        ggplot,
        labs,
        theme,
        theme_minimal,
        geom_vline,
        geom_point,
        scale_x_log10,
        facet_wrap,
        element_text,
        element_blank,
        geom_boxplot,
        geom_errorbar,
        coord_cartesian,
        scale_colour_manual,
        scale_fill_manual,
    )
    return (
        Path,
        aes,
        coord_cartesian,
        element_text,
        facet_wrap,
        geom_boxplot,
        geom_errorbar,
        geom_point,
        ggplot,
        json,
        labs,
        mo,
        np,
        pd,
        scale_colour_manual,
        scale_fill_manual,
        theme,
        theme_minimal,
    )


@app.cell
def _():
    return


@app.cell
def _(Path):
    results_folder = Path("data/results")
    ppdoc_results_folder = results_folder / "results_ppdoc"
    torch_results_folder = results_folder / "results_pytorch"

    doclaynet_json = "doclaynet/benchmark_all_results.json"
    HNLA_json = "hnla2013/benchmark_all_results.json"
    NCSE_json = "ncse/benchmark_all_results.json"
    HNLA_nosssu_json = "hnla2013_unique_ssu/benchmark_all_results.json"
    NCSE_nossu_json = "ncse_unique_ssu/benchmark_all_results.json"
    # The Spiritualist is scored in fakenewslib (notebooks/mote_cer_real_models.py), checked out
    # next to this repo
    spiritualist_folder = Path("../fakenewslib/data/positional_uncertainty")
    return (
        HNLA_json,
        HNLA_nosssu_json,
        NCSE_json,
        NCSE_nossu_json,
        doclaynet_json,
        ppdoc_results_folder,
        spiritualist_folder,
        torch_results_folder,
    )


@app.function
def harmonise_mote_columns(df):
    """Fill in MOTe columns for results saved before the MOTe reformulation, and the
    old COTe columns for results saved after it, so either vintage loads the same.

    With unit weights MOTe = 1 - COTe and Missing = 1 - Coverage exactly.
    """
    if "mote" not in df and "cot_score" in df:
        df["mote"] = 1 - df["cot_score"]
    if "cot_score" not in df and "mote" in df:
        df["cot_score"] = 1 - df["mote"]
    if "missing" not in df and "coverage" in df:
        df["missing"] = 1 - df["coverage"]
    if "coverage" not in df and "missing" in df:
        df["coverage"] = 1 - df["missing"]
    return df


@app.cell
def _(json, np, pd, ppdoc_results_folder, torch_results_folder):

    def create_results_table(filename):
        MODEL_NAME_MAP = {
            "DocLayout-YOLO": "YOLO",
            "DoclingLayoutHeron": "Heron",
            "PPDocLayout-L": "PPDoc-L",
            "PPDocLayout-M": "PPDoc-M",
            "PPDocLayout-S": "PPDoc-S",
        }

        out_df = []
        for _folder in [ppdoc_results_folder, torch_results_folder]:
            # Not every dataset is benchmarked in both environments (e.g. Spiritualist)
            if not (_folder / filename).exists():
                continue
            with open(_folder / filename, "r") as f:
                _data = json.load(f)
            _df = pd.DataFrame.from_dict(_data["models"], orient="index")
            out_df.append(_df)
        out_df = harmonise_mote_columns(pd.concat(out_df, ignore_index=False))
        out_df = out_df.loc[
            :, [
                "mote",
                "missing",
                "overlap",
                "trespass",
                "excess",
                "mean_iou",
                "map",
                "f1_50",
            ]
        ]
        out_df.index = out_df.index.map(lambda x: MODEL_NAME_MAP.get(x, x))
        out_df = out_df.sort_index()
        return out_df

    def df_to_latex_md(df, caption, label, cer=None):
        """cer: per-model page-mean CER bounds (cer_lo, cer_hi); if None they come from the mean M, O, T."""
        col_rename = {
            "mote": "MOTe",
            "missing": "Missing",
            "overlap": "Overlap",
            "trespass": "Trespass",
            "excess": "Excess",
            "mean_iou": "IoU",
            "map": "mAP",
            "f1_50": "F1@50",
        }

        col_best = {
            "mote": "low",
            "missing": "low",
            "overlap": "low",
            "trespass": "low",
            "excess": "low",
            "mean_iou": "high",
            "map": "high",
            "f1_50": "high",
        }

        from latex_tables import bold_best_cols, latex_table

        lower_cols = [col_rename[c] for c, d in col_best.items() if d == "low"]
        df_fmt = bold_best_cols(
            df.astype(float).rename(columns=col_rename).rename_axis("Model"),
            lower_cols=lower_cols,
            fmt="{:.2f}",
        )
        # Predicted CER range [max(M, O), M+O+2T]
        if cer is None:
            _m, _o, _t = (df[k].astype(float) for k in ["missing", "overlap", "trespass"])
            _cer_lo, _cer_hi = np.maximum(_m, _o), _m + _o + 2 * _t
        else:
            _cer = cer.set_axis(cer.index.astype(str)).reindex(df.index)
            _cer_lo, _cer_hi = _cer["cer_lo"], _cer["cer_hi"]
        df_fmt["P-CER"] = [f"{lo:.2f}--{hi:.2f}" for lo, hi in zip(_cer_lo, _cer_hi)]
        caption = f"{caption}. P-CER is the estimated parsing-induced CER."
        latex = latex_table(df_fmt, caption=caption, label=label, echo=False)

        return f"### LaTeX Table Output\n```latex\n{latex}\n```"
    return create_results_table, df_to_latex_md


@app.cell
def _(
    HNLA_json,
    HNLA_nosssu_json,
    NCSE_json,
    NCSE_nossu_json,
    create_results_table,
    doclaynet_json,
):
    doclaynet_df = create_results_table(doclaynet_json)
    HNLA2013_df = create_results_table(HNLA_json)
    NCSE_df = create_results_table(NCSE_json)

    HNLA2013_nossu_df = create_results_table(HNLA_nosssu_json)
    NCSE_nossu_df = create_results_table(NCSE_nossu_json)
    return HNLA2013_df, HNLA2013_nossu_df, NCSE_df, NCSE_nossu_df, doclaynet_df


@app.cell
def _(HNLA2013_nossu_df):
    HNLA2013_nossu_df
    return


@app.cell
def _(NCSE_nossu_df):
    NCSE_nossu_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SSU vs no SSU

    Only Trespass depends on the SSU grouping (Missing, Overlap and Excess are
    identical under both ground truths), so only Trespass is compared here.
    $\Delta$ is no SSU minus SSU.
    """)
    return


@app.cell
def _(pd):
    def ssu_comparison_table(ssu_df, nossu_df):
        return pd.DataFrame(
            {
                "SSU": ssu_df["trespass"],
                "No SSU": nossu_df["trespass"],
                r"Difference": nossu_df["trespass"] - ssu_df["trespass"],
            }
        ).astype(float)

    def ssu_comparison_latex_md(df, caption, label):
        from latex_tables import bold_best_cols, latex_table

        lower_cols = ["SSU", "No SSU", r"Difference"]
        df_fmt = bold_best_cols(df.rename_axis("Model"), lower_cols=lower_cols, fmt="{:.2f}")
        latex = latex_table(df_fmt, caption=caption, label=label, echo=False)
        return f"### LaTeX Table Output\n```latex\n{latex}\n```"
    return ssu_comparison_latex_md, ssu_comparison_table


@app.cell
def _(
    HNLA2013_df,
    HNLA2013_nossu_df,
    NCSE_df,
    NCSE_nossu_df,
    ssu_comparison_table,
):
    NCSE_ssu_comparison_df = ssu_comparison_table(NCSE_df, NCSE_nossu_df)
    HNLA2013_ssu_comparison_df = ssu_comparison_table(HNLA2013_df, HNLA2013_nossu_df)
    NCSE_ssu_comparison_df
    return HNLA2013_ssu_comparison_df, NCSE_ssu_comparison_df


@app.cell
def _(HNLA2013_ssu_comparison_df):
    HNLA2013_ssu_comparison_df
    return


@app.cell
def _(NCSE_ssu_comparison_df, mo, ssu_comparison_latex_md):
    mo.md(
        ssu_comparison_latex_md(
            NCSE_ssu_comparison_df,
            caption="Trespass for NCSE with and without SSU grouping",
            label="tab:ssu_ncse",
        )
    )
    return


@app.cell
def _(HNLA2013_ssu_comparison_df, mo, ssu_comparison_latex_md):
    mo.md(
        ssu_comparison_latex_md(
            HNLA2013_ssu_comparison_df,
            caption="Trespass for HNLA2013 with and without SSU grouping",
            label="tab:ssu_hnla2013",
        )
    )
    return


@app.cell
def _(HNLA2013_df, NCSE_df, doclaynet_df, pd):
    all_datasets = pd.concat(
        [
            doclaynet_df.assign(dataset="doclaynet"),
            HNLA2013_df.assign(dataset="HNLA2013"),
            NCSE_df.assign(dataset="NCSE"),
        ],
        ignore_index=False,
    )
    all_datasets = all_datasets.reset_index()
    return (all_datasets,)


@app.cell
def _(aes, all_datasets, geom_point, ggplot):
    (
        ggplot(all_datasets, aes(x="mote", y="mean_iou", colour="index", shape="dataset"))
        + geom_point()
    ).draw()
    return


@app.cell
def _(doclaynet_df):
    doclaynet_df.index
    return


@app.cell
def _():
    return


@app.cell
def _(df_to_latex_md, doclaynet_df, mo, page_cer_mean_df):
    mo.md(
        df_to_latex_md(
            doclaynet_df,
            caption="Model results for DocLayNet",
            label="tab:res_doclaynet",
            cer=page_cer_mean_df.loc["DocLayNet"],
        )
    )
    return


@app.cell
def _(HNLA2013_df, df_to_latex_md, mo, page_cer_mean_df):
    mo.md(
        df_to_latex_md(
            HNLA2013_df,
            caption="Model results for HNLA2013",
            label="tab:res_HNLA2013",
            cer=page_cer_mean_df.loc["HNLA2013"],
        )
    )
    return


@app.cell
def _(NCSE_df, df_to_latex_md, mo, page_cer_mean_df):
    mo.md(
        df_to_latex_md(
            NCSE_df,
            caption="Model results for NCSE",
            label="tab:res_ncse",
            cer=page_cer_mean_df.loc["NCSE"],
        )
    )
    return


@app.cell
def _(np, pd, spiritualist_folder):
    # Per-page scores against the GT lines grouped into article SSUs, and per-model mAP and
    # F1@50 pooled over every box, as for the other datasets. Written by fakenewslib's
    # notebooks/mote_cer_real_models.py
    _model_names = {
        "DocLayout-YOLO": "YOLO",
        "DoclingLayoutHeron": "Heron",
        "PPDocLayout-L": "PPDoc-L",
        "PPDocLayout-M": "PPDoc-M",
        "PPDocLayout-S": "PPDoc-S",
    }
    spiritualist_pages_df = pd.read_parquet(spiritualist_folder / "mote_cer_spirit_results.parquet")
    spiritualist_pooled_df = pd.read_parquet(spiritualist_folder / "mote_cer_spirit_pooled.parquet")
    for _df in [spiritualist_pages_df, spiritualist_pooled_df]:
        _df["model"] = _df["model"].map(lambda x: _model_names.get(x, x))
    # Predicted CER bounds per page: [max(M, O), M+O+2T]
    spiritualist_pages_df["cer_lo"] = np.maximum(
        spiritualist_pages_df["missing"], spiritualist_pages_df["overlap"]
    )
    spiritualist_pages_df["cer_hi"] = (
        spiritualist_pages_df["missing"]
        + spiritualist_pages_df["overlap"]
        + 2 * spiritualist_pages_df["trespass"]
    )
    return spiritualist_pages_df, spiritualist_pooled_df


@app.cell
def _(df_to_latex_md, mo, spiritualist_pages_df, spiritualist_pooled_df):
    # Kept out of the cross-dataset analyses below apart from the box plots
    _means = spiritualist_pages_df.groupby("model")[
        ["mote", "missing", "overlap", "trespass", "excess", "mean_iou", "cer_lo", "cer_hi"]
    ].mean()
    spiritualist_df = _means.drop(columns=["cer_lo", "cer_hi"]).join(
        spiritualist_pooled_df.set_index("model")[["map", "f1_50"]]
    ).sort_index()
    mo.vstack(
        [
            spiritualist_df,
            mo.md(
                df_to_latex_md(
                    spiritualist_df,
                    caption="Model results for the Spiritualist",
                    label="tab:res_spiritualist",
                    cer=_means[["cer_lo", "cer_hi"]],
                )
            ),
        ]
    )
    return


@app.cell
def _(Path, json, pd):

    def load_results_to_dataframe(base_path: str) -> pd.DataFrame:
        """
        Load all JSON result files from a directory and return a combined DataFrame.

        Each row contains the model name, filename, and per-image metrics.
        Files without a 'model' field are skipped.

        Args:
            base_path: Path to directory containing JSON result files

        Returns:
            DataFrame with columns: model, filename, mean_iou, coverage, missing,
            overlap, trespass, excess, cot_score, mote
        """
        records = []
        path = Path(base_path)

        for json_file in path.glob("*.json"):
            with open(json_file, "r") as f:
                data = json.load(f)

            model = data.get("model")
            if not model:
                continue

            for image_result in data.get("per_image_results", []):
                record = {
                    "model": model,
                    "filename": image_result.get("filename"),
                    **image_result.get("metrics", {}),
                }
                records.append(record)

        return harmonise_mote_columns(pd.DataFrame(records))
    return (load_results_to_dataframe,)


@app.cell
def _(load_results_to_dataframe, pd):
    MODEL_NAME_MAP = {
        "juliozhao/DocLayout-YOLO-DocStructBench": "YOLO",
        "docling-project/docling-layout-heron": "Heron",
        "PP-DocLayout-L": "PPDoc-L",
        "PP-DocLayout-M": "PPDoc-M",
        "PP-DocLayout-S": "PPDoc-S",
    }

    NCSE_comparison_df = pd.concat(
        [
            load_results_to_dataframe("data/results/results_ppdoc/ncse"),
            load_results_to_dataframe("data/results/results_pytorch/ncse"),
        ]
    )

    NCSE_comparison_df["model"] = NCSE_comparison_df["model"].map(
        lambda x: MODEL_NAME_MAP.get(x, x)
    )

    HNLA_comparison_df = pd.concat(
        [
            load_results_to_dataframe("data/results/results_ppdoc/hnla2013"),
            load_results_to_dataframe("data/results/results_pytorch/hnla2013"),
        ]
    )

    HNLA_comparison_df["model"] = HNLA_comparison_df["model"].map(
        lambda x: MODEL_NAME_MAP.get(x, x)
    )

    doclaynet_comparison_df = pd.concat(
        [
            load_results_to_dataframe("data/results/results_ppdoc/doclaynet"),
            load_results_to_dataframe("data/results/results_pytorch/doclaynet"),
        ]
    )

    doclaynet_comparison_df["model"] = doclaynet_comparison_df["model"].map(
        lambda x: MODEL_NAME_MAP.get(x, x)
    )
    return (
        HNLA_comparison_df,
        MODEL_NAME_MAP,
        NCSE_comparison_df,
        doclaynet_comparison_df,
    )


@app.cell
def _(NCSE_comparison_df, aes, geom_point, ggplot):
    (ggplot(NCSE_comparison_df, aes(y="missing", x="overlap", color="model")) + geom_point()).draw()
    return


@app.cell
def _(NCSE_comparison_df, aes, geom_point, ggplot):
    (ggplot(NCSE_comparison_df, aes(y="missing", x="trespass", color="model")) + geom_point()).draw()
    return


@app.cell
def _(NCSE_comparison_df):
    NCSE_comparison_df.drop(columns=["filename", "model"]).corr()
    return


@app.cell
def _(NCSE_comparison_df):
    NCSE_comparison_df.drop(columns="filename").groupby("model").apply(
        lambda x: x.drop(columns="model").corr()
    )
    return


@app.cell
def _(NCSE_comparison_df):
    NCSE_comparison_df.drop(columns=["filename", "model"]).corr(method="spearman")
    return


@app.cell
def _(HNLA_comparison_df):
    HNLA_comparison_df.drop(columns=["filename", "model"]).corr(method="spearman")
    return


@app.cell
def _(doclaynet_comparison_df):
    doclaynet_comparison_df.drop(columns=["filename", "model"]).corr(method="spearman")
    return


@app.cell
def _(NCSE_comparison_df, aes, geom_point, ggplot):
    (ggplot(NCSE_comparison_df, aes(x="mote", y="mean_iou", colour="model")) + geom_point()).draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distribution of page-level MOTe and F1

    Per-page MOTe and F1@50 for every model, one panel per dataset (SSU ground truth). F1@50
    is scored per page with the same COCO matching as the pooled F1@50 in the results tables.
    """)
    return


@app.cell
def _(Path, pd):
    def compute_page_map(cache_path=Path("data/results/per_page_map.csv")):
        """Per-page mAP@[.50:.95] (class-agnostic COCO) from the prediction CSVs, cached.

        The benchmark JSONs only store mAP for the whole dataset. This scores each page
        on its own with the same MAPMetric the runner uses, so pooling these pages
        reproduces the stored dataset-level mAP. Pages with no GT are undefined (NaN).
        """
        if cache_path.exists():
            return pd.read_csv(cache_path)
        from mote_error.map_metric import MAPMetric

        csv_model_names = {
            "DocLayout-YOLO": "YOLO",
            "DoclingLayoutHeron": "Heron",
            "PPDocLayout-L": "PPDoc-L",
            "PPDocLayout-M": "PPDoc-M",
            "PPDocLayout-S": "PPDoc-S",
        }
        dataset_names = {"doclaynet": "DocLayNet", "hnla2013": "HNLA2013", "ncse": "NCSE"}

        def to_records(df):
            # mAP in the benchmark ignores class, so every box is the same class here too
            return [
                {"x": r.x, "y": r.y, "width": r.width, "height": r.height,
                 "class": "object", "confidence": r.confidence}
                for r in df.itertuples()
            ]

        rows = []
        for folder, dataset in dataset_names.items():
            for csv_path in sorted(Path("data/results").glob(f"results_*/{folder}/*_predictions.csv")):
                preds_df = pd.read_csv(csv_path)
                for (model, filename), page_df in preds_df.groupby(["model", "filename"], sort=False):
                    metric = MAPMetric()
                    metric.update(
                        to_records(page_df[page_df["source"] == "pred"]),
                        to_records(page_df[page_df["source"] == "gt"]),
                    )
                    page_map = metric.compute()["map"]
                    rows.append({
                        "dataset": dataset,
                        "model": csv_model_names.get(model, model),
                        "filename": filename,
                        "map": page_map if page_map >= 0 else float("nan"),
                    })
        out = pd.DataFrame(rows)
        out.to_csv(cache_path, index=False)
        return out

    page_map_df = compute_page_map()
    return (page_map_df,)


@app.cell
def _(
    HNLA_comparison_df,
    NCSE_comparison_df,
    doclaynet_comparison_df,
    np,
    page_map_df,
    pd,
):
    MODEL_ORDER = ["YOLO", "Heron", "PPDoc-L", "PPDoc-M", "PPDoc-S"]
    DATASET_ORDER = ["DocLayNet", "HNLA2013", "NCSE"]
    # Fixed categorical order so each model keeps its colour across every figure
    MODEL_COLOURS = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))

    page_mote_df = pd.concat(
        [
            doclaynet_comparison_df.assign(dataset="DocLayNet"),
            HNLA_comparison_df.assign(dataset="HNLA2013"),
            NCSE_comparison_df.assign(dataset="NCSE"),
        ],
        ignore_index=True,
    )[
        [
            "dataset", "model", "filename", "mote", "missing", "overlap",
            "trespass", "excess", "mean_iou", "f1_50",
        ]
    ]
    page_mote_df = page_mote_df.merge(page_map_df, on=["dataset", "model", "filename"], how="left")
    # Predicted CER bounds per page: [max(M, O), M+O+2T]
    page_mote_df["cer_lo"] = np.maximum(page_mote_df["missing"], page_mote_df["overlap"])
    page_mote_df["cer_hi"] = (
        page_mote_df["missing"] + page_mote_df["overlap"] + 2 * page_mote_df["trespass"]
    )
    page_mote_df["dataset"] = pd.Categorical(page_mote_df["dataset"], DATASET_ORDER)
    page_mote_df["model"] = pd.Categorical(page_mote_df["model"], MODEL_ORDER)
    page_cer_mean_df = page_mote_df.groupby(["dataset", "model"], observed=True)[
        ["cer_lo", "cer_hi"]
    ].mean()
    return MODEL_COLOURS, MODEL_ORDER, page_cer_mean_df, page_mote_df


@app.cell
def _(
    MODEL_COLOURS,
    MODEL_ORDER,
    Path,
    aes,
    coord_cartesian,
    element_text,
    facet_wrap,
    geom_boxplot,
    geom_errorbar,
    ggplot,
    labs,
    page_mote_df,
    pd,
    scale_fill_manual,
    spiritualist_pages_df,
    theme,
    theme_minimal,
):
    _box_df = pd.concat(
        [
            page_mote_df[["dataset", "model", "mote", "f1_50"]].astype({"dataset": str, "model": str}),
            spiritualist_pages_df[["model", "mote", "f1_50"]].assign(dataset="Spiritualist"),
        ],
        ignore_index=True,
    )
    _box_df["dataset"] = pd.Categorical(
        _box_df["dataset"], ["Spiritualist", "NCSE", "HNLA2013", "DocLayNet"]
    )
    _box_df["model"] = pd.Categorical(_box_df["model"], MODEL_ORDER)

    def score_boxplot(score_col, y_label, filename, y_max=None):
        """Box plot of `score_col` per model, one panel per dataset, with the mean as a dashed
        line; saved to data/figures. y_max zooms the axis without dropping any pages from the
        boxes or means."""
        means = (
            _box_df.groupby(["dataset", "model"], observed=True)[score_col].mean().rename("mean").reset_index()
        )
        plot = (
            ggplot(_box_df, aes(x="model", y=score_col, fill="model"))
            + geom_boxplot(outlier_size=0.8, outlier_alpha=0.5)
            + geom_errorbar(
                aes(x="model", ymin="mean", ymax="mean"),
                data=means,
                inherit_aes=False,
                width=0.75,
                linetype="dashed",
                size=0.7,
            )
            + facet_wrap("~dataset", ncol=2)
            + scale_fill_manual(values=MODEL_COLOURS)
            + labs(x="Model", y=y_label)
            + theme_minimal()
            + theme(
                figure_size=(8, 6),
                legend_position="none",
                strip_text=element_text(size=11, weight="bold"),
            )
        )
        if y_max is not None:
            plot += coord_cartesian(ylim=(0, y_max))
        out_dir = Path("data/figures/analysing_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        plot.save(out_dir / filename, dpi=300, verbose=False)
        # Return a matplotlib figure: marimo does not display plotnine's own PNG output
        return plot.draw()

    return (score_boxplot,)


@app.cell
def _(score_boxplot):
    # The long DocLayNet tail (up to ~3.6) is cut off by the zoom, not removed from the boxes
    score_boxplot("mote", "Page-level MOTe", "mote_boxplot_by_dataset.png", y_max=1.5)
    return


@app.cell
def _(score_boxplot):
    score_boxplot("f1_50", "Page-level F1@50", "f1_boxplot_by_dataset.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ratio of predictions to ground truth regions

    Per page, ratio $= n_\text{pred} / n_\text{GT}$ (SSU ground truth, GT counted as individual
    boxes). Pages with no GT are dropped, as the ratio is undefined. The dataset-level ratio
    pools every page: total predictions over total GT boxes.

    A ratio of 1 is ideal and both directions are errors, so the raw ratio can only pick up
    one direction (typically under-prediction). $|\log_2 \text{ratio}|$ measures the distance
    from 1 in either direction, so twice and half as many boxes count equally; a page with
    no predictions has an infinite distance, which Spearman simply ranks last.
    """)
    return


@app.cell
def _(MODEL_ORDER, Path, np, page_mote_df, pd):
    _csv_model_names = {
        "DocLayout-YOLO": "YOLO",
        "DoclingLayoutHeron": "Heron",
        "PPDocLayout-L": "PPDoc-L",
        "PPDocLayout-M": "PPDoc-M",
        "PPDocLayout-S": "PPDoc-S",
    }
    _counts = []
    for _folder, _dataset in {"doclaynet": "DocLayNet", "hnla2013": "HNLA2013", "ncse": "NCSE"}.items():
        for _csv_path in sorted(Path("data/results").glob(f"results_*/{_folder}/*_predictions.csv")):
            _c = (
                pd.read_csv(_csv_path, usecols=["model", "filename", "source"])
                .groupby(["model", "filename", "source"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["pred", "gt"], fill_value=0)
                .rename(columns={"pred": "n_pred", "gt": "n_gt"})
                .reset_index()
            )
            _c["model"] = _c["model"].map(lambda x: _csv_model_names.get(x, x))
            _counts.append(_c.assign(dataset=_dataset))
    _counts = pd.concat(_counts, ignore_index=True)

    # Pages with neither GT nor predictions have no rows in the CSVs; inner join drops
    # them along with the pages that only lack GT (filtered below)
    pred_gt_df = page_mote_df.astype({"dataset": str, "model": str}).merge(
        _counts, on=["dataset", "model", "filename"], how="inner"
    )
    pred_gt_df = pred_gt_df[pred_gt_df["n_gt"] > 0].copy()
    pred_gt_df["ratio"] = pred_gt_df["n_pred"] / pred_gt_df["n_gt"]
    with np.errstate(divide="ignore"):
        pred_gt_df["abs_log2_ratio"] = np.abs(np.log2(pred_gt_df["ratio"]))

    # Dataset-level ratio per model: pooled over pages, plus the typical page
    pred_gt_ratio_df = (
        pred_gt_df.groupby(["dataset", "model"])
        .agg(n_pred=("n_pred", "sum"), n_gt=("n_gt", "sum"), median_page=("ratio", "median"))
        .assign(pooled=lambda d: d["n_pred"] / d["n_gt"])
    )
    pred_gt_ratio_table = pred_gt_ratio_df[["pooled", "median_page"]].unstack("dataset")
    pred_gt_ratio_table = pred_gt_ratio_table.swaplevel(axis=1).sort_index(axis=1).loc[MODEL_ORDER]
    pred_gt_ratio_table.round(2)
    return pred_gt_df, pred_gt_ratio_df


@app.cell
def _(pd, pred_gt_df):
    RATIO_CORR_METRICS = {
        "mote": "MOTe",
        "missing": "Missing",
        "overlap": "Overlap",
        "trespass": "Trespass",
        "excess": "Excess",
        "mean_iou": "IoU",
        "map": "mAP",
        "f1_50": "F1@50",
    }

    def ratio_spearman(df):
        """Spearman of ratio and |log2 ratio| against each metric, pages pooled over models."""
        _corr = (
            df[["ratio", "abs_log2_ratio", *RATIO_CORR_METRICS]]
            .corr(method="spearman")
            .loc[["ratio", "abs_log2_ratio"], list(RATIO_CORR_METRICS)]
            .rename(columns=RATIO_CORR_METRICS)
        )
        return _corr

    # mAP is NaN on no-GT pages, but those are already dropped, so every metric uses the same pages
    ratio_spearman_df = pd.concat(
        {_name: ratio_spearman(_df) for _name, _df in pred_gt_df.groupby("dataset")},
        names=["Dataset", "Ratio"],
    ).rename(index={"ratio": "Ratio", "abs_log2_ratio": r"$|\log_2 \text{Ratio}|$"})
    ratio_spearman_df.round(2)
    return RATIO_CORR_METRICS, ratio_spearman, ratio_spearman_df


@app.cell
def _(ratio_spearman_df):
    ratio_spearman_df
    return


@app.cell
def _(mo, ratio_spearman_df):
    from latex_tables import latex_table as _latex_table

    # Only the |log2 ratio| rows, so one row per dataset and no Ratio column
    _fmt = (
        ratio_spearman_df.xs(r"$|\log_2 \text{Ratio}|$", level="Ratio")
        .drop(columns="Excess")
        .map("{:.2f}".format)
    )
    _latex = _latex_table(
        _fmt,
        caption=(
            r"Page-level Spearman correlation between $|\log_2 \text{Ratio}|$ of predicted to "
            "ground truth regions and each metric, per dataset (pages pooled over all five models)"
        ),
        label="tab:pred_gt_ratio_spearman",
        echo=False,
    )
    mo.md(f"### LaTeX Table Output\n```latex\n{_latex}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Without PPDoc-M and PPDoc-S

    PPDoc-M and PPDoc-S barely predict anything on HNLA2013 (ratio $\approx 0.04$) and
    under-predict on NCSE, so the pooled correlations above may mostly separate "predicts"
    from "does not predict". Same table using only YOLO, Heron and PPDoc-L.
    """)
    return


@app.cell
def _(mo, pd, pred_gt_df, ratio_spearman):
    from latex_tables import latex_table as _latex_table

    STRONG_MODELS = ["YOLO", "Heron", "PPDoc-L"]
    ratio_spearman_strong_df = pd.concat(
        {
            _name: ratio_spearman(_df)
            for _name, _df in pred_gt_df[pred_gt_df["model"].isin(STRONG_MODELS)].groupby("dataset")
        },
        names=["Dataset", "Ratio"],
    ).rename(index={"ratio": "Ratio", "abs_log2_ratio": r"$|\log_2 \text{Ratio}|$"})

    _fmt = ratio_spearman_strong_df.xs(r"$|\log_2 \text{Ratio}|$", level="Ratio").map(
        "{:.2f}".format
    )
    _latex = _latex_table(
        _fmt,
        caption=(
            r"Page-level Spearman correlation between $|\log_2 \text{Ratio}|$ of predicted to "
            "ground truth regions and each metric, per dataset (pages pooled over YOLO, Heron "
            "and PPDoc-L)"
        ),
        label="tab:pred_gt_ratio_spearman_strong",
        echo=False,
    )
    mo.vstack(
        [ratio_spearman_strong_df.round(2), mo.md(f"### LaTeX Table Output\n```latex\n{_latex}\n```")]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dataset level ($n=5$ models per dataset)

    Pooled ratio for each model against its dataset-level scores. With $n=5$ only
    $|\rho| = 1$ reaches $p < 0.05$, so treat these as indicative.
    """)
    return


@app.cell
def _(
    HNLA2013_df,
    NCSE_df,
    RATIO_CORR_METRICS,
    doclaynet_df,
    np,
    pd,
    pred_gt_ratio_df,
):
    _rows = {}
    for _name, _df in [("DocLayNet", doclaynet_df), ("HNLA2013", HNLA2013_df), ("NCSE", NCSE_df)]:
        _d = _df[list(RATIO_CORR_METRICS)].astype(float).join(
            pred_gt_ratio_df.loc[_name, "pooled"].rename("ratio")
        )
        _d["abs_log2_ratio"] = np.abs(np.log2(_d["ratio"]))
        _c = _d.corr(method="spearman").rename(columns=RATIO_CORR_METRICS)
        _rows[(_name, "Ratio")] = _c.loc["ratio", list(RATIO_CORR_METRICS.values())]
        _rows[(_name, r"$|\log_2 \text{Ratio}|$")] = _c.loc[
            "abs_log2_ratio", list(RATIO_CORR_METRICS.values())
        ]
    dataset_ratio_spearman_df = pd.DataFrame(_rows).T.rename_axis(["Dataset", "Ratio"])
    dataset_ratio_spearman_df.round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # No ssu
    """)
    return


@app.cell
def _(MODEL_NAME_MAP, load_results_to_dataframe, pd):
    NCSE_nossu_comparison_df = pd.concat(
        [
            load_results_to_dataframe("data/results/results_ppdoc/ncse_unique_ssu"),
            load_results_to_dataframe("data/results/results_pytorch/ncse_unique_ssu"),
        ]
    )

    NCSE_nossu_comparison_df["model"] = NCSE_nossu_comparison_df["model"].map(
        lambda x: MODEL_NAME_MAP.get(x, x)
    )
    return (NCSE_nossu_comparison_df,)


@app.cell
def _(NCSE_nossu_comparison_df):
    NCSE_nossu_comparison_df.drop(columns=["filename", "model"]).corr()
    return


@app.cell
def _(MODEL_NAME_MAP, load_results_to_dataframe, pd):
    HNLA_nossu_comparison_df = pd.concat(
        [
            load_results_to_dataframe("data/results/results_ppdoc/hnla2013_unique_ssu"),
            load_results_to_dataframe("data/results/results_pytorch/hnla2013_unique_ssu"),
        ]
    )

    HNLA_nossu_comparison_df["model"] = HNLA_nossu_comparison_df["model"].map(
        lambda x: MODEL_NAME_MAP.get(x, x)
    )

    HNLA_nossu_comparison_df.drop(columns=["filename", "model"]).corr()
    return


if __name__ == "__main__":
    app.run()
