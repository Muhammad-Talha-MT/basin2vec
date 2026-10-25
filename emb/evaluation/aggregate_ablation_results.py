#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_VARIANTS = [
    "instance_contrastive",
    "meta_as_positive",
    "metadata_concat",
    "no_metadata_encoder",
    "no_meta_denominator",
    "no_monthly_branch",
    "no_negative_margin",
    "no_static_branch",
]


def parse_csv_list(text, cast=str):
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate Basin2Vec Round-2 temporal/spatial ablation results."
    )
    p.add_argument("--root", default=str(SCRIPT_DIR / "evaluation_outputs" / "ablation_round2"))
    p.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated ablation variants.",
    )
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Default: <root>/summary",
    )
    return p.parse_args()


def select_one(df: pd.DataFrame, *, split: str) -> pd.Series:
    d = df.copy()

    if "model_name" in d.columns:
        d = d[d["model_name"].astype(str).str.lower() == "lstm"]
    if "static_source" in d.columns:
        d = d[d["static_source"].astype(str).str.lower() == "basin2vec"]
    if "seq_len" in d.columns:
        d = d[pd.to_numeric(d["seq_len"], errors="coerce") == 365]
    if "horizon" in d.columns:
        d = d[pd.to_numeric(d["horizon"], errors="coerce") == 1]
    if "lead" in d.columns:
        d = d[pd.to_numeric(d["lead"], errors="coerce") == 0]

    if len(d) != 1:
        raise RuntimeError(
            f"Expected exactly one LSTM/Basin2Vec/L365/H1/lead0 row for {split}; "
            f"found {len(d)}."
        )
    return d.iloc[0]


def load_temporal(run_dir: Path):
    candidates = [
        run_dir / "ALL_CAMELS_temporal_per_basin_summary.csv",
        run_dir / "PAPER_TABLE_CAMELS_temporal_per_basin_summary.csv",
        run_dir / "ALL_CAMELS_temporal_per_basin_summary_partial.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None

    row = select_one(pd.read_csv(path), split="temporal")
    return {
        "temporal_median_NSE": float(row["median_NSE"]),
        "temporal_median_KGE": float(row["median_KGE"]),
        "temporal_n_basins": int(row.get("n_basins", 0)),
        "temporal_summary_file": str(path),
    }


def load_spatial_fold_summary(run_dir: Path):
    # Preserve the supplied spatial script's original aggregation:
    # average of the six fold-level basin medians.
    path = run_dir / "KFOLD_AGGREGATED_CAMELS671_per_basin_summary.csv"
    if path.exists():
        row = select_one(pd.read_csv(path), split="spatial-fold-aggregate")
        out = {
            "spatial_foldmean_median_NSE": float(
                row["median_NSE_mean_across_folds"]
            ),
            "spatial_foldmean_median_KGE": float(
                row["median_KGE_mean_across_folds"]
            ),
            "spatial_n_folds": int(row.get("n_folds", 0)),
            "spatial_fold_summary_file": str(path),
        }
        if "median_NSE_std_across_folds" in row.index:
            out["spatial_within_seed_fold_std_NSE"] = float(
                row["median_NSE_std_across_folds"]
            )
        if "median_KGE_std_across_folds" in row.index:
            out["spatial_within_seed_fold_std_KGE"] = float(
                row["median_KGE_std_across_folds"]
            )
        return out

    # Fallback: calculate from fold summary rows.
    raw = run_dir / "ALL_CAMELS671_per_basin_summary_all_models_all_folds.csv"
    if not raw.exists():
        return None
    df = pd.read_csv(raw)
    if "model_name" in df.columns:
        df = df[df["model_name"].astype(str).str.lower() == "lstm"]
    if "static_source" in df.columns:
        df = df[df["static_source"].astype(str).str.lower() == "basin2vec"]
    if "seq_len" in df.columns:
        df = df[pd.to_numeric(df["seq_len"], errors="coerce") == 365]
    if "horizon" in df.columns:
        df = df[pd.to_numeric(df["horizon"], errors="coerce") == 1]
    if "lead" in df.columns:
        df = df[pd.to_numeric(df["lead"], errors="coerce") == 0]
    if len(df) == 0:
        return None

    return {
        "spatial_foldmean_median_NSE": float(np.nanmean(df["median_NSE"])),
        "spatial_foldmean_median_KGE": float(np.nanmean(df["median_KGE"])),
        "spatial_within_seed_fold_std_NSE": float(np.nanstd(df["median_NSE"], ddof=1)),
        "spatial_within_seed_fold_std_KGE": float(np.nanstd(df["median_KGE"], ddof=1)),
        "spatial_n_folds": int(df["fold_id"].nunique()),
        "spatial_fold_summary_file": str(raw),
    }


def load_spatial_pooled(run_dir: Path):
    """
    Pool per-basin metrics from the six held-out folds.

    Each basin should appear in exactly one outer test fold. This produces the
    literal median across all held-out basins, alongside the original script's
    mean-of-fold-medians statistic.
    """
    files = sorted(
        run_dir.glob(
            "fold*_lstm_basin2vec_seq365_lead0_horizon1_camels671_kratzert_optionA/"
            "test_per_basin_metrics_horizon1.csv"
        )
    )
    if not files:
        return None

    frames = []
    for p in files:
        d = pd.read_csv(p)
        if "lead" in d.columns:
            d = d[pd.to_numeric(d["lead"], errors="coerce") == 0]
        d["_source_file"] = str(p)
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    if len(df) == 0:
        return None

    # Validate outer-CV uniqueness. A basin must be a held-out test basin once.
    counts = df.groupby("site_id").size()
    duplicated = counts[counts > 1]
    if len(duplicated):
        raise RuntimeError(
            f"{run_dir}: {len(duplicated)} basins occur more than once in pooled "
            "outer-fold per-basin metrics."
        )

    nse = pd.to_numeric(df["NSE"], errors="coerce")
    kge = pd.to_numeric(df["KGE"], errors="coerce")

    return {
        "spatial_pooled_median_NSE": float(np.nanmedian(nse)),
        "spatial_pooled_median_KGE": float(np.nanmedian(kge)),
        "spatial_pooled_mean_NSE": float(np.nanmean(nse)),
        "spatial_pooled_mean_KGE": float(np.nanmean(kge)),
        "spatial_pooled_n_basins": int(df["site_id"].nunique()),
        "spatial_pooled_n_files": int(len(files)),
    }


def mean_std_table(seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "temporal_median_NSE",
        "temporal_median_KGE",
        "spatial_foldmean_median_NSE",
        "spatial_foldmean_median_KGE",
        "spatial_pooled_median_NSE",
        "spatial_pooled_median_KGE",
    ]

    rows = []
    for variant, d in seed_df.groupby("variant", sort=False):
        row = {
            "variant": variant,
            "n_seeds": int(d["embedding_seed"].nunique()),
            "seeds": ",".join(map(str, sorted(d["embedding_seed"].unique()))),
        }
        for c in metric_cols:
            if c not in d.columns:
                continue
            vals = pd.to_numeric(d[c], errors="coerce")
            vals = vals[np.isfinite(vals)]
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{c}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{c}_max"] = float(vals.max()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        def fmt(mean_col, std_col):
            m = r.get(mean_col, np.nan)
            s = r.get(std_col, np.nan)
            if not np.isfinite(m):
                return ""
            if np.isfinite(s):
                return f"{m:.3f} ± {s:.3f}"
            return f"{m:.3f}"

        rows.append({
            "Variant": r["variant"],
            "Temporal NSE": fmt(
                "temporal_median_NSE_mean",
                "temporal_median_NSE_std",
            ),
            "Temporal KGE": fmt(
                "temporal_median_KGE_mean",
                "temporal_median_KGE_std",
            ),
            # Original supplied spatial code: mean of fold-level medians.
            "Spatial NSE (fold mean)": fmt(
                "spatial_foldmean_median_NSE_mean",
                "spatial_foldmean_median_NSE_std",
            ),
            "Spatial KGE (fold mean)": fmt(
                "spatial_foldmean_median_KGE_mean",
                "spatial_foldmean_median_KGE_std",
            ),
            # Also provide pooled outer-fold medians for protocol auditing.
            "Spatial NSE (pooled basins)": fmt(
                "spatial_pooled_median_NSE_mean",
                "spatial_pooled_median_NSE_std",
            ),
            "Spatial KGE (pooled basins)": fmt(
                "spatial_pooled_median_KGE_mean",
                "spatial_pooled_median_KGE_std",
            ),
            "Seeds": r.get("n_seeds", np.nan),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_csv_list(args.variants, str)
    seeds = parse_csv_list(args.seeds, int)

    rows = []
    missing = []

    for variant in variants:
        for seed in seeds:
            row = {
                "variant": variant,
                "embedding_seed": int(seed),
                "downstream_seed": int(seed),
            }

            temporal_dir = root / "temporal" / variant / f"seed_{seed}"
            spatial_dir = root / "spatial" / variant / f"seed_{seed}"

            t = load_temporal(temporal_dir)
            sf = load_spatial_fold_summary(spatial_dir)
            sp = load_spatial_pooled(spatial_dir)

            if t:
                row.update(t)
            else:
                missing.append({
                    "variant": variant,
                    "seed": seed,
                    "split": "temporal",
                    "path": str(temporal_dir),
                })

            if sf:
                row.update(sf)
            else:
                missing.append({
                    "variant": variant,
                    "seed": seed,
                    "split": "spatial_fold_summary",
                    "path": str(spatial_dir),
                })

            if sp:
                row.update(sp)
            else:
                missing.append({
                    "variant": variant,
                    "seed": seed,
                    "split": "spatial_pooled",
                    "path": str(spatial_dir),
                })

            rows.append(row)

    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(out_dir / "ROUND2_ABLATION_seed_level_summary.csv", index=False)

    summary = mean_std_table(seed_df)
    summary.to_csv(out_dir / "ROUND2_ABLATION_mean_std_summary.csv", index=False)

    paper = make_paper_table(summary)
    paper.to_csv(out_dir / "ROUND2_ABLATION_paper_table.csv", index=False)

    missing_file = out_dir / "ROUND2_ABLATION_missing_results.csv"
    if missing:
        pd.DataFrame(missing).to_csv(missing_file, index=False)
    elif missing_file.exists():
        missing_file.unlink()

    with open(out_dir / "README_summary.json", "w") as f:
        json.dump({
            "root": str(root),
            "variants": variants,
            "seeds": seeds,
            "statistical_unit_across_runs": (
                "Each seed is an independent pretrained Basin2Vec checkpoint and "
                "uses the same numeric seed for downstream LSTM initialization."
            ),
            "temporal_primary": (
                "Median NSE/KGE across temporal evaluation basins for each seed; "
                "then mean ± sample standard deviation across seeds."
            ),
            "spatial_foldmean": (
                "Matches the supplied spatial evaluator's KFOLD_AGGREGATED output: "
                "mean of the six outer-fold median basin metrics within each seed; "
                "then mean ± sample standard deviation across seeds."
            ),
            "spatial_pooled": (
                "Additional audit statistic: pool each basin's outer-fold test "
                "metric exactly once and take the median across all held-out basins."
            ),
        }, f, indent=2)

    print("\nSeed-level results:")
    print(seed_df.to_string(index=False))
    print("\nAcross-seed mean ± std:")
    print(paper.to_string(index=False))
    print(f"\nSaved summary files to: {out_dir}")


if __name__ == "__main__":
    main()
