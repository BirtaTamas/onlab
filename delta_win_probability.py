import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


LABEL_COL = "ct_win"
ID_COLUMNS = ("round_num", "tick", "seconds_in_round")
UTILITY_TOKENS = (
    "utility",
    "smoke",
    "flash",
    "he",
    "molly",
    "inferno",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delta win probability elemzes with-utility es no-utility modellek kozott."
    )
    parser.add_argument(
        "--with-run-dir",
        type=str,
        required=True,
        help="With-utility modellfutas mappa.",
    )
    parser.add_argument(
        "--no-run-dir",
        type=str,
        required=True,
        help="No-utility modellfutas mappa.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Kimeneti mappa. Alapbol a with-run-dir alatt delta_win_probability.",
    )
    parser.add_argument("--split", choices=["valid", "test", "all"], default="test")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument(
        "--max-csvs",
        type=int,
        default=None,
        help="Opcionális limit gyors/eszettanulmany futashoz.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "best_metrics.json"
    return load_json(metrics_path)


def model_path_for(run_dir: Path) -> Path:
    for name in ("xgboost_model.json", "best_xgboost_model.json"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Nem talaltam modell json fajlt itt: {run_dir}")


def read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readline().strip().split(",")


def infer_data_root(run_dir: Path) -> Path:
    # run_dir = onlab/artifacts/modellfutasok/<run>
    return run_dir.parents[2] / "processed_full"


def manifest_paths(run_dir: Path, split: str, max_csvs: int | None) -> List[Tuple[str, Path]]:
    manifest_path = run_dir / "sampled_split_manifest.csv"
    data_root = infer_data_root(run_dir)
    rows: List[Tuple[str, Path]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split != "all" and row["split"] != split:
                continue
            rel_csv = row["csv_path"]
            rows.append((rel_csv, data_root.parent / rel_csv))
            if max_csvs is not None and len(rows) >= max_csvs:
                break
    return rows


def is_utility_feature(name: str) -> bool:
    lowered = name.lower()
    tokens = set(lowered.split("_"))
    return any(token in tokens for token in UTILITY_TOKENS)


def utility_activity_columns(header: Iterable[str]) -> List[str]:
    cols = []
    for col in header:
        if col == LABEL_COL or col in ID_COLUMNS:
            continue
        if is_utility_feature(col):
            cols.append(col)
    return cols


def binary_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    eps = 1e-15
    p = np.clip(y_prob, eps, 1.0 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def predict_booster(
    booster: xgb.Booster,
    x_data: pd.DataFrame,
) -> np.ndarray:
    best_iteration = getattr(booster, "best_iteration", None)
    dmat = xgb.DMatrix(x_data)
    if best_iteration is not None and best_iteration >= 0:
        return booster.predict(dmat, iteration_range=(0, int(best_iteration) + 1))
    return booster.predict(dmat)


def stream_prediction_rows(
    csv_items: List[Tuple[str, Path]],
    with_features: List[str],
    no_features: List[str],
    with_model: xgb.Booster,
    no_model: xgb.Booster,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    needed_base = set(with_features) | set(no_features) | {LABEL_COL} | set(ID_COLUMNS)

    for rel_csv, csv_path in csv_items:
        header = read_header(csv_path)
        utility_cols = utility_activity_columns(header)
        usecols = [col for col in header if col in needed_base or col in utility_cols]
        reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size)
        row_offset = 0
        for chunk in reader:
            n = len(chunk)
            y_true = chunk[LABEL_COL].fillna(0).astype(np.int64).to_numpy()

            x_with = chunk.reindex(columns=with_features, fill_value=0).fillna(0)
            x_no = chunk.reindex(columns=no_features, fill_value=0).fillna(0)
            p_with = predict_booster(with_model, x_with)
            p_no = predict_booster(no_model, x_no)

            delta = p_with - p_no
            abs_delta = np.abs(delta)
            good_direction = np.where(y_true == 1, delta > 0, delta < 0)
            ll_with = binary_logloss(y_true, p_with)
            ll_no = binary_logloss(y_true, p_no)
            logloss_improvement = ll_no - ll_with

            if utility_cols:
                utility_abs_sum = chunk.reindex(columns=utility_cols, fill_value=0).fillna(0).abs().sum(axis=1)
                utility_active = utility_abs_sum.to_numpy() > 0
            else:
                utility_abs_sum = pd.Series(np.zeros(n), index=chunk.index)
                utility_active = np.zeros(n, dtype=bool)

            out = pd.DataFrame(
                {
                    "csv_path": rel_csv,
                    "csv_row_index": np.arange(row_offset, row_offset + n),
                    "round_num": chunk["round_num"].to_numpy() if "round_num" in chunk else np.nan,
                    "tick": chunk["tick"].to_numpy() if "tick" in chunk else np.nan,
                    "seconds_in_round": chunk["seconds_in_round"].to_numpy()
                    if "seconds_in_round" in chunk
                    else np.nan,
                    "ct_win": y_true,
                    "p_with_utility": p_with,
                    "p_no_utility": p_no,
                    "delta": delta,
                    "abs_delta": abs_delta,
                    "good_direction": good_direction,
                    "logloss_with": ll_with,
                    "logloss_no": ll_no,
                    "logloss_improvement": logloss_improvement,
                    "utility_active": utility_active,
                    "utility_abs_sum": utility_abs_sum.to_numpy(),
                }
            )
            yield out
            row_offset += n


def summarize_rows(df: pd.DataFrame, prefix: str) -> Dict[str, object]:
    if df.empty:
        return {
            f"{prefix}_rows": 0,
        }
    return {
        f"{prefix}_rows": int(len(df)),
        f"{prefix}_mean_abs_delta": float(df["abs_delta"].mean()),
        f"{prefix}_median_abs_delta": float(df["abs_delta"].median()),
        f"{prefix}_p90_abs_delta": float(df["abs_delta"].quantile(0.90)),
        f"{prefix}_p95_abs_delta": float(df["abs_delta"].quantile(0.95)),
        f"{prefix}_p99_abs_delta": float(df["abs_delta"].quantile(0.99)),
        f"{prefix}_mean_delta": float(df["delta"].mean()),
        f"{prefix}_good_direction_rate": float(df["good_direction"].mean()),
        f"{prefix}_mean_logloss_improvement": float(df["logloss_improvement"].mean()),
        f"{prefix}_positive_logloss_improvement_rate": float((df["logloss_improvement"] > 0).mean()),
    }


def write_summary(
    output_dir: Path,
    split: str,
    all_rows: pd.DataFrame,
    top_rows: pd.DataFrame,
    with_metrics: dict,
    no_metrics: dict,
) -> None:
    utility_active = all_rows[all_rows["utility_active"]]
    utility_inactive = all_rows[~all_rows["utility_active"]]

    summary = {
        "split": split,
        "with_run": {
            "feature_count": with_metrics.get("feature_count"),
            "test": with_metrics.get("test"),
            "valid": with_metrics.get("valid"),
        },
        "no_run": {
            "feature_count": no_metrics.get("feature_count"),
            "test": no_metrics.get("test"),
            "valid": no_metrics.get("valid"),
        },
        **summarize_rows(all_rows, "all"),
        **summarize_rows(utility_active, "utility_active"),
        **summarize_rows(utility_inactive, "utility_inactive"),
    }

    per_csv = (
        all_rows.groupby("csv_path")
        .agg(
            rows=("csv_path", "size"),
            mean_abs_delta=("abs_delta", "mean"),
            p95_abs_delta=("abs_delta", lambda s: s.quantile(0.95)),
            good_direction_rate=("good_direction", "mean"),
            mean_logloss_improvement=("logloss_improvement", "mean"),
            utility_active_rate=("utility_active", "mean"),
        )
        .reset_index()
        .sort_values("mean_abs_delta", ascending=False)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "delta_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    top_rows.to_csv(output_dir / "top_delta_rows.csv", index=False)
    per_csv.to_csv(output_dir / "per_csv_delta_summary.csv", index=False)

    lines = [
        "# Delta Win Probability Summary",
        "",
        f"- split: `{split}`",
        f"- rows: `{summary['all_rows']}`",
        f"- mean abs delta: `{summary['all_mean_abs_delta']:.6f}`",
        f"- p95 abs delta: `{summary['all_p95_abs_delta']:.6f}`",
        f"- good direction rate: `{summary['all_good_direction_rate']:.4f}`",
        f"- mean logloss improvement: `{summary['all_mean_logloss_improvement']:.6f}`",
        "",
        "## Utility Active Rows",
        "",
        f"- rows: `{summary.get('utility_active_rows', 0)}`",
    ]
    if summary.get("utility_active_rows", 0):
        lines.extend(
            [
                f"- mean abs delta: `{summary['utility_active_mean_abs_delta']:.6f}`",
                f"- p95 abs delta: `{summary['utility_active_p95_abs_delta']:.6f}`",
                f"- good direction rate: `{summary['utility_active_good_direction_rate']:.4f}`",
                f"- mean logloss improvement: `{summary['utility_active_mean_logloss_improvement']:.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Utility Inactive Rows",
            "",
            f"- rows: `{summary.get('utility_inactive_rows', 0)}`",
        ]
    )
    if summary.get("utility_inactive_rows", 0):
        lines.extend(
            [
                f"- mean abs delta: `{summary['utility_inactive_mean_abs_delta']:.6f}`",
                f"- p95 abs delta: `{summary['utility_inactive_p95_abs_delta']:.6f}`",
                f"- good direction rate: `{summary['utility_inactive_good_direction_rate']:.4f}`",
                f"- mean logloss improvement: `{summary['utility_inactive_mean_logloss_improvement']:.6f}`",
            ]
        )
    lines.extend(["", "## Top CSVs By Mean Abs Delta", ""])
    for _, row in per_csv.head(20).iterrows():
        lines.append(
            f"- `{row['csv_path']}` mean_abs_delta=`{row['mean_abs_delta']:.6f}` "
            f"good_direction_rate=`{row['good_direction_rate']:.4f}` "
            f"utility_active_rate=`{row['utility_active_rate']:.4f}`"
        )
    (output_dir / "delta_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    with_run_dir = Path(args.with_run_dir)
    no_run_dir = Path(args.no_run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else with_run_dir / "delta_win_probability"

    with_metrics = load_metrics(with_run_dir)
    no_metrics = load_metrics(no_run_dir)
    with_features = with_metrics["features"]
    no_features = no_metrics["features"]

    with_model = xgb.Booster()
    with_model.load_model(str(model_path_for(with_run_dir)))
    no_model = xgb.Booster()
    no_model.load_model(str(model_path_for(no_run_dir)))

    csv_items = manifest_paths(with_run_dir, split=args.split, max_csvs=args.max_csvs)
    print(f"Split: {args.split}, CSV count: {len(csv_items)}")
    print(f"With features: {len(with_features)}, no features: {len(no_features)}")

    chunks = []
    top_rows = pd.DataFrame()
    for index, rows in enumerate(
        stream_prediction_rows(
            csv_items=csv_items,
            with_features=with_features,
            no_features=no_features,
            with_model=with_model,
            no_model=no_model,
            chunk_size=args.chunk_size,
        ),
        start=1,
    ):
        if index % 25 == 0 or index == 1:
            print(f"Processed chunks: {index}")
        chunks.append(rows)
        candidate_top = pd.concat([top_rows, rows], ignore_index=True)
        top_rows = candidate_top.nlargest(args.top_k, "abs_delta")

    all_rows = pd.concat(chunks, ignore_index=True)
    top_rows = top_rows.sort_values("abs_delta", ascending=False)
    write_summary(
        output_dir=output_dir,
        split=args.split,
        all_rows=all_rows,
        top_rows=top_rows,
        with_metrics=with_metrics,
        no_metrics=no_metrics,
    )
    print(f"Delta elemzes mentve: {output_dir}")


if __name__ == "__main__":
    main()
