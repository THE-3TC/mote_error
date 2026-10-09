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
    import patchworklib as pw
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
        geom_density,
        scale_colour_manual,
        scale_fill_manual,
    )
    return (
        Path,
        aes,
        element_text,
        facet_wrap,
        geom_density,
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
    return (
        HNLA_json,
        HNLA_nosssu_json,
        NCSE_json,
        NCSE_nossu_json,
        doclaynet_json,
        ppdoc_results_folder,
        torch_results_folder,
    )


@app.cell
def _(json, pd, ppdoc_results_folder, torch_results_folder):

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
            with open(_folder / filename, "r") as f:
                _data = json.load(f)
            _df = pd.DataFrame.from_dict(_data["models"], orient="index")
            out_df.append(_df)
        out_df = pd.concat(out_df, ignore_index=False)
        out_df = out_df.loc[
            :, [
                "cot_score",
                "coverage",
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

    def df_to_latex_md(df, caption, label):
        col_rename = {
            "cot_score": "COTe",
            "coverage": "Coverage",
            "overlap": "Overlap",
            "trespass": "Trespass",
            "excess": "Excess",
            "mean_iou": "IoU",
            "map": "mAP",
            "f1_50": "F1@50",
        }

        col_best = {
            "cot_score": "high",
            "coverage": "high",
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

    Only Trespass depends on the SSU grouping (Coverage, Overlap and Excess are
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Combined SSU table (NCSE and HNLA2013)

    One table for both datasets. Bold marks the best value within each dataset.
    """)
    return


@app.cell
def _(HNLA2013_ssu_comparison_df, NCSE_ssu_comparison_df, mo, pd):
    from latex_tables import bold_best_cols as _bold_best_cols, latex_table as _latex_table

    _cols = ["SSU", "No SSU", r"Difference"]
    _blocks = []
    for _dataset, _df in [("NCSE", NCSE_ssu_comparison_df), ("HNLA2013", HNLA2013_ssu_comparison_df)]:
        # Bold per dataset block, not across the whole column
        _fmt = _bold_best_cols(
            _df[_cols], lower_cols=_cols, fmt="{:.2f}"
        ).rename_axis("Model").reset_index()
        _fmt.insert(0, "Dataset", [_dataset] + [""] * (len(_fmt) - 1))
        _blocks.append(_fmt)
    ssu_combined_df = pd.concat(_blocks, ignore_index=True).set_index("Dataset")

    _latex = _latex_table(
        ssu_combined_df,
        caption=(
            "Trespass with and without SSU grouping of the ground truth. "
            r"$\Delta$ Trespass is no SSU minus SSU. Coverage, Overlap and Excess "
            "are unaffected by the grouping."
        ),
        label="tab:ssu_impact",
        echo=False,
    ).replace("\nHNLA2013 &", "\n\\midrule\nHNLA2013 &")
    mo.vstack([ssu_combined_df, mo.md(f"### LaTeX Table Output\n```latex\n{_latex}\n```")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset-level Spearman correlation between metrics

    Each observation is one model's dataset-level score (5 models per dataset, SSU ground
    truth). Per dataset this is only $n=5$, so a single swapped pair moves $\rho$ by 0.1 and
    only $|\rho| = 1$ reaches $p < 0.05$; the pooled value uses all $n=15$ model-dataset pairs.
    """)
    return


@app.cell
def _(HNLA2013_df, NCSE_df, doclaynet_df, pd):
    CORR_METRICS = {"cot_score": "COTe", "mean_iou": "IoU", "f1_50": "F1@50", "map": "mAP"}

    dataset_level_df = pd.concat(
        [
            doclaynet_df.assign(dataset="DocLayNet"),
            HNLA2013_df.assign(dataset="HNLA2013"),
            NCSE_df.assign(dataset="NCSE"),
        ]
    ).rename(columns=CORR_METRICS)[["dataset", *CORR_METRICS.values()]]

    # Pooled matrix across all 15 model-dataset pairs
    pooled_spearman_df = (
        dataset_level_df[list(CORR_METRICS.values())].astype(float).corr(method="spearman")
    )

    # COTe against each other metric, per dataset and pooled
    _rows = {
        _name: _df[list(CORR_METRICS.values())].astype(float).corr(method="spearman").loc["COTe"]
        for _name, _df in [*dataset_level_df.groupby("dataset"), ("All", dataset_level_df)]
    }
    cote_spearman_df = pd.DataFrame(_rows).T.drop(columns="COTe").rename_axis("Dataset")
    cote_spearman_df.columns = [f"COTe vs {_c}" for _c in cote_spearman_df.columns]
    pooled_spearman_df
    return cote_spearman_df, pooled_spearman_df


@app.cell
def _(cote_spearman_df):
    cote_spearman_df
    return


@app.cell
def _(cote_spearman_df, mo, pooled_spearman_df):
    from latex_tables import latex_table as _latex_table

    _pooled = pooled_spearman_df.map("{:.2f}".format).rename_axis("")
    _pooled_latex = _latex_table(
        _pooled,
        caption=(
            "Spearman correlation between dataset-level metrics, pooled over all "
            "model-dataset pairs ($n=15$)"
        ),
        label="tab:spearman_pooled",
        echo=False,
    )
    _per_dataset = cote_spearman_df.map("{:.2f}".format)
    _per_dataset_latex = _latex_table(
        _per_dataset,
        caption=(
            "Spearman correlation between dataset-level COTe and the other metrics, per "
            "dataset ($n=5$ models) and pooled ($n=15$)"
        ),
        label="tab:spearman_cote",
        echo=False,
    )
    mo.md(
        f"### LaTeX Table Output\n```latex\n{_pooled_latex}\n```\n\n"
        f"```latex\n{_per_dataset_latex}\n```"
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
        ggplot(all_datasets, aes(x="cot_score", y="mean_iou", colour="index", shape="dataset"))
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
def _(df_to_latex_md, doclaynet_df, mo):
    mo.md(
        df_to_latex_md(
            doclaynet_df, caption="Model results for DocLayNet", label="tab:res_doclaynet"
        )
    )
    return


@app.cell
def _(HNLA2013_df, df_to_latex_md, mo):
    mo.md(
        df_to_latex_md(HNLA2013_df, caption="Model results for HNLA2013", label="tab:res_HNLA2013")
    )
    return


@app.cell
def _(NCSE_df, df_to_latex_md, mo):
    mo.md(df_to_latex_md(NCSE_df, caption="Model results for NCSE", label="tab:res_ncse"))
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
            DataFrame with columns: model, filename, mean_iou, coverage,
            overlap, trespass, excess, cot_score
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

        return pd.DataFrame(records)
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
    (ggplot(NCSE_comparison_df, aes(y="coverage", x="overlap", color="model")) + geom_point()).draw()
    return


@app.cell
def _(NCSE_comparison_df, aes, geom_point, ggplot):
    (ggplot(NCSE_comparison_df, aes(y="coverage", x="trespass", color="model")) + geom_point()).draw()
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
    (ggplot(NCSE_comparison_df, aes(x="cot_score", y="mean_iou", colour="model")) + geom_point()).draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distribution of page-level COTe

    Density of per-page COTe for every model, one panel per dataset (SSU ground truth),
    and the mean vs median per model. If the two disagree in model ranking, the headline
    (mean) results are being driven by a tail of very good or very bad pages rather than
    typical performance.

    Each is shown twice: on the raw scores, and with COTe floored at 0. A page scoring 0
    is equivalent to making no predictions at all, so the floor stops pages where the
    predictions are worse than nothing from dominating the mean.
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
        from cotescore.map_metric import MAPMetric

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
    page_map_df,
    pd,
):
    MODEL_ORDER = ["YOLO", "Heron", "PPDoc-L", "PPDoc-M", "PPDoc-S"]
    DATASET_ORDER = ["DocLayNet", "HNLA2013", "NCSE"]
    # Fixed categorical order so each model keeps its colour across every figure
    MODEL_COLOURS = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))

    page_cote_df = pd.concat(
        [
            doclaynet_comparison_df.assign(dataset="DocLayNet"),
            HNLA_comparison_df.assign(dataset="HNLA2013"),
            NCSE_comparison_df.assign(dataset="NCSE"),
        ],
        ignore_index=True,
    )[
        [
            "dataset", "model", "filename", "cot_score", "coverage", "overlap",
            "trespass", "excess", "mean_iou", "f1_50",
        ]
    ]
    page_cote_df = page_cote_df.merge(page_map_df, on=["dataset", "model", "filename"], how="left")
    page_cote_df["cot_score_floored"] = page_cote_df["cot_score"].clip(lower=0)
    page_cote_df["dataset"] = pd.Categorical(page_cote_df["dataset"], DATASET_ORDER)
    page_cote_df["model"] = pd.Categorical(page_cote_df["model"], MODEL_ORDER)
    return MODEL_COLOURS, MODEL_ORDER, page_cote_df


@app.cell
def _(
    MODEL_COLOURS,
    MODEL_ORDER,
    Path,
    aes,
    element_text,
    facet_wrap,
    geom_density,
    ggplot,
    labs,
    mo,
    page_cote_df,
    scale_colour_manual,
    scale_fill_manual,
    theme,
    theme_minimal,
):
    def cote_density_plot(score_col, x_label, filename):
        """Density of `score_col` per model, one panel per dataset; saved to data/figures."""
        plot = (
            ggplot(page_cote_df, aes(x=score_col, colour="model"))
            + geom_density(alpha=0.15, size=0.7)
            + facet_wrap("~dataset", ncol=1, scales="free_y")
            + scale_colour_manual(values=MODEL_COLOURS)
            + scale_fill_manual(values=MODEL_COLOURS)
            + labs(x=x_label, y="Density", colour="Model", fill="Model")
            + theme_minimal()
            + theme(
                figure_size=(6, 8),
                legend_position="bottom",
                strip_text=element_text(size=11, weight="bold"),
            )
        )
        out_dir = Path("data/figures/analysing_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        plot.save(out_dir / filename, dpi=300, verbose=False)
        # Return a matplotlib figure: marimo does not display plotnine's own PNG output
        return plot.draw()

    def mean_median_table(score_col):
        """Mean, median, their gap and the model rank under each, per dataset and model."""
        df = (
            page_cote_df.groupby(["dataset", "model"], observed=True)[score_col]
            .agg(Mean="mean", Median="median")
            .reset_index()
        )
        df["Mean $-$ Median"] = df["Mean"] - df["Median"]
        for col in ["Mean", "Median"]:
            df[f"Rank ({col.lower()})"] = (
                df.groupby("dataset", observed=True)[col]
                .rank(ascending=False, method="min")
                .astype(int)
            )
        df["model"] = df["model"].astype(str)
        return df.sort_values(
            ["dataset", "model"], key=lambda c: c.map(MODEL_ORDER.index) if c.name == "model" else c
        ).rename(columns={"dataset": "Dataset", "model": "Model"})

    def mean_median_latex_md(df, caption, label):
        from latex_tables import latex_table

        df = df.copy()
        for col in ["Mean", "Median", "Mean $-$ Median"]:
            df[col] = df[col].map("{:.2f}".format)
        # Print each dataset name once, on the first row of its block
        df["Dataset"] = df["Dataset"].astype(str).where(~df["Dataset"].duplicated(), "")
        latex = latex_table(df.set_index("Dataset"), caption=caption, label=label, echo=False)
        return mo.md(f"### LaTeX Table Output\n```latex\n{latex}\n```")
    return cote_density_plot, mean_median_latex_md, mean_median_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw page-level COTe
    """)
    return


@app.cell
def _(cote_density_plot):
    cote_density_plot("cot_score", "Page-level COTe", "cote_density_by_dataset.png")
    return


@app.cell
def _(mean_median_table):
    cote_mean_median_df = mean_median_table("cot_score")
    cote_mean_median_df
    return (cote_mean_median_df,)


@app.cell
def _(cote_mean_median_df, mean_median_latex_md):
    mean_median_latex_md(
        cote_mean_median_df,
        caption="Mean and median page-level COTe per model and dataset",
        label="tab:cote_mean_median",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Page-level COTe floored at 0
    """)
    return


@app.cell
def _(cote_density_plot):
    cote_density_plot(
        "cot_score_floored",
        "Page-level COTe (floored at 0)",
        "cote_floored_density_by_dataset.png",
    )
    return


@app.cell
def _(mean_median_table):
    cote_floored_mean_median_df = mean_median_table("cot_score_floored")
    cote_floored_mean_median_df
    return (cote_floored_mean_median_df,)


@app.cell
def _(cote_floored_mean_median_df, mean_median_latex_md):
    mean_median_latex_md(
        cote_floored_mean_median_df,
        caption="Mean and median page-level COTe per model and dataset, with COTe floored at 0",
        label="tab:cote_floored_mean_median",
    )
    return


@app.cell
def _(page_cote_df):
    # Share of pages each model scores below 0, i.e. the pages the floor changes
    (
        page_cote_df.assign(below_zero=page_cote_df["cot_score"] < 0)
        .groupby(["dataset", "model"], observed=True)["below_zero"]
        .mean()
        .unstack("dataset")
        .round(3)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mean vs median for IoU, F1@50 and mAP

    The same comparison for the other metrics, all per page. mAP is computed per page from
    the prediction CSVs (class-agnostic, as in the benchmark); its mean over pages is not the
    dataset-level mAP in the results tables, which pools every box before ranking. Likewise
    the F1@50 mean is the page mean (`f1_50_page_mean`), not the pooled F1@50.
    """)
    return


@app.cell
def _(mean_median_table):
    # mAP is undefined (NaN) on the 5 DocLayNet pages with no ground truth; they are skipped
    OTHER_METRICS = {"mean_iou": "IoU", "f1_50": "F1@50", "map": "mAP"}
    metric_mean_median = {
        _label: mean_median_table(_col) for _col, _label in OTHER_METRICS.items()
    }
    return (metric_mean_median,)


@app.cell
def _(metric_mean_median):
    metric_mean_median["IoU"]
    return


@app.cell
def _(metric_mean_median):
    metric_mean_median["F1@50"]
    return


@app.cell
def _(metric_mean_median):
    metric_mean_median["mAP"]
    return


@app.cell
def _(mean_median_latex_md, metric_mean_median, mo):
    mo.vstack(
        [
            mean_median_latex_md(
                _df,
                caption=f"Mean and median page-level {_label} per model and dataset",
                label=f"tab:{_label.lower().replace('@', '')}_mean_median",
            )
            for _label, _df in metric_mean_median.items()
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary: mean $-$ median for every metric

    One row per model and dataset; a $^\dagger$ marks where the model's rank differs
    between mean and median.
    """)
    return


@app.cell
def _(MODEL_ORDER, cote_mean_median_df, metric_mean_median, mo, pd):
    from latex_tables import latex_table as _latex_table

    _all = {"COTe": cote_mean_median_df, **metric_mean_median}
    _gap = {}
    for _label, _df in _all.items():
        _d = _df.set_index(["Dataset", "Model"])
        _flag = (_d["Rank (mean)"] != _d["Rank (median)"]).map({True: "$^\\dagger$", False: ""})
        _gap[_label] = _d["Mean $-$ Median"].map("{:.2f}".format) + _flag
    mean_median_gap_df = pd.DataFrame(_gap).reset_index()
    mean_median_gap_df = mean_median_gap_df.sort_values(
        ["Dataset", "Model"],
        key=lambda c: c.map(MODEL_ORDER.index) if c.name == "Model" else c.astype(str),
    )
    _fmt = mean_median_gap_df.copy()
    _fmt["Dataset"] = _fmt["Dataset"].astype(str).where(~_fmt["Dataset"].duplicated(), "")
    _latex = _latex_table(
        _fmt.set_index("Dataset"),
        caption=(
            "Mean minus median of page-level scores per model and dataset. "
            "$^\\dagger$ marks a model whose rank within the dataset differs between mean and median."
        ),
        label="tab:mean_median_gap",
        echo=False,
    )
    mo.vstack([mean_median_gap_df, mo.md(f"### LaTeX Table Output\n```latex\n{_latex}\n```")])
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
def _(MODEL_ORDER, Path, np, page_cote_df, pd):
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
    pred_gt_df = page_cote_df.astype({"dataset": str, "model": str}).merge(
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
        "cot_score": "COTe",
        "coverage": "Coverage",
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
