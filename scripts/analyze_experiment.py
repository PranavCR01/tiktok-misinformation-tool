# scripts/analyze_experiment.py

import re
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import dedent


# ---------- helpers ----------
def _safe_col(df, name, fallback=None):
    return df[name] if name in df.columns else fallback


def _split_keywords(cell):
    if not isinstance(cell, str) or not cell.strip():
        return []
    # allow ; or , as separators
    parts = re.split(r"[;,]", cell)
    return [p.strip() for p in parts if p.strip()]


def _save_bar_plot(series, title, xlabel, ylabel, out_path, rotate=False):
    plt.figure()
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_hist(values, title, xlabel, out_path, bins=10):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp-dir",
        default="experiments/exp-001-mistral-baseline",
        help="Path to an experiment folder that contains results.csv",
    )
    ap.add_argument("--top-k", type=int, default=20, help="Top-K keywords bar chart")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    csv_path = exp_dir / "results.csv"
    assert csv_path.exists(), f"results.csv not found at: {csv_path}"

    # load
    df = pd.read_csv(csv_path)

    # basic meta
    n = len(df)
    prompt_id = (
        _safe_col(df, "prompt_id", "").unique().tolist() if "prompt_id" in df else []
    )
    model_name = (
        _safe_col(df, "model_name", "").unique().tolist() if "model_name" in df else []
    )

    # label distribution
    label_counts = df["label"].value_counts().sort_index()
    label_png = exp_dir / "label_distribution.png"
    _save_bar_plot(
        label_counts, "Label Distribution", "Label", "Count", label_png, rotate=False
    )

    # confidence histogram
    conf = pd.to_numeric(
        _safe_col(df, "confidence_score", pd.Series([])), errors="coerce"
    ).dropna()
    conf_png = exp_dir / "confidence_hist.png"
    if not conf.empty:
        _save_hist(
            conf,
            "Confidence Scores",
            "confidence",
            conf_png,
            bins=min(10, max(5, len(conf) // 2)),
        )

    # latency histogram
    lat = pd.to_numeric(
        _safe_col(df, "time_taken_sec", pd.Series([])), errors="coerce"
    ).dropna()
    lat_png = exp_dir / "latency_hist.png"
    if not lat.empty:
        _save_hist(
            lat,
            "Model Latency (seconds)",
            "seconds",
            lat_png,
            bins=min(10, max(5, len(lat) // 2)),
        )

    # keywords frequency
    kw_series = (
        df["keywords"].apply(_split_keywords)
        if "keywords" in df.columns
        else pd.Series([[]] * n)
    )
    all_kws = [k for row in kw_series for k in row]
    kw_freq = pd.Series(all_kws).value_counts().head(args.top_k)
    kw_csv = exp_dir / "keywords_top.csv"
    if not kw_freq.empty:
        kw_freq.rename_axis("keyword").reset_index(name="count").to_csv(
            kw_csv, index=False
        )
        kw_png = exp_dir / "keywords_top.png"
        _save_bar_plot(
            kw_freq,
            f"Top {args.top_k} Keywords",
            "keyword",
            "count",
            kw_png,
            rotate=True,
        )
    else:
        kw_png = None

    # simple stats
    stats = {
        "num_rows": n,
        "unique_videos": (
            df["video_file"].nunique() if "video_file" in df.columns else n
        ),
        "labels": ", ".join([f"{k}:{v}" for k, v in label_counts.to_dict().items()]),
        "avg_confidence": round(float(conf.mean()), 3) if not conf.empty else "NA",
        "avg_latency_sec": round(float(lat.mean()), 2) if not lat.empty else "NA",
        "min_latency_sec": round(float(lat.min()), 2) if not lat.empty else "NA",
        "max_latency_sec": round(float(lat.max()), 2) if not lat.empty else "NA",
    }

    # write README.md
    readme_path = exp_dir / "README.md"
    prompt_txt = ", ".join(map(str, prompt_id)) if prompt_id else "(unknown)"
    model_txt = ", ".join(map(str, model_name)) if model_name else "(unknown)"

    md = dedent(
        f"""
    # Experiment report

    **Folder:** `{exp_dir.as_posix()}`

    - **Model:** {model_txt}
    - **Prompt:** {prompt_txt}
    - **Rows:** {stats['num_rows']}
    - **Unique videos:** {stats['unique_videos']}

    ## Quick stats
    - Label distribution: {stats['labels']}
    - Avg confidence: **{stats['avg_confidence']}**
    - Latency (sec) — avg: **{stats['avg_latency_sec']}**, min: {stats['min_latency_sec']}, max: {stats['max_latency_sec']}

    ## Plots
    ![Label Distribution](./{label_png.name})
    {"![Confidence Histogram](./" + conf_png.name + ")" if conf_png.exists() else ""}
    {"![Latency Histogram](./" + lat_png.name + ")" if lat_png.exists() else ""}
    {"![Top Keywords](./" + kw_png.name + ")" if kw_png and Path(kw_png).exists() else ""}

    ## Files
    - `results.csv` — raw outputs
    - `{label_png.name}` — label counts
    - `{conf_png.name}` — confidence histogram
    - `{lat_png.name}` — latency histogram
    - `keywords_top.csv` — top keyword counts
    {("- `" + (kw_png.name if kw_png else "") + "` — top keyword bar chart") if kw_png and Path(kw_png).exists() else ""}
    """
    )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md.strip() + "\n")

    print(f"Wrote plots + README to: {exp_dir}")


if __name__ == "__main__":
    main()
