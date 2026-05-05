import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


LABEL_COL = "ct_win"
ID_COLUMNS = ("round_num", "tick", "seconds_in_round")
UNCERTAINTY_BINS = (
    ("very_uncertain_0_02", 0.00, 0.02),
    ("uncertain_0_05", 0.02, 0.05),
    ("mild_uncertain_0_10", 0.05, 0.10),
    ("medium_0_20", 0.10, 0.20),
    ("confident_gt_0_20", 0.20, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Megnezi, hogy a utility foleg a no-utility modell bizonytalan snapshotjain mozdit-e."
    )
    parser.add_argument("--with-run-dir", type=str, required=True)
    parser.add_argument("--no-run-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", choices=["valid", "test", "all"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--max-csvs", type=int, default=None)
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


def infer_data_root(run_dir: Path) -> Path:
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


def predict_booster(booster: xgb.Booster, x_data: pd.DataFrame) -> np.ndarray:
    best_iteration = getattr(booster, "best_iteration", None)
    dmat = xgb.DMatrix(x_data)
    if best_iteration is not None and best_iteration >= 0:
        return booster.predict(dmat, iteration_range=(0, int(best_iteration) + 1))
    return booster.predict(dmat)


def binary_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    eps = 1e-15
    p = np.clip(y_prob, eps, 1.0 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def bin_name_for(margin: np.ndarray) -> np.ndarray:
    out = np.empty(len(margin), dtype=object)
    for name, low, high in UNCERTAINTY_BINS:
        mask = (margin >= low) & (margin < high)
        out[mask] = name
    return out


def stream_rows(
    csv_items: List[Tuple[str, Path]],
    with_features: List[str],
    no_features: List[str],
    with_model: xgb.Booster,
    no_model: xgb.Booster,
    threshold: float,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    needed = set(with_features) | set(no_features) | {LABEL_COL} | set(ID_COLUMNS)

    for rel_csv, csv_path in csv_items:
        with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline().strip().split(",")
        usecols = [col for col in header if col in needed]

        row_offset = 0
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
            n = len(chunk)
            y_true = chunk[LABEL_COL].fillna(0).astype(np.int64).to_numpy()

            x_with = chunk.reindex(columns=with_features, fill_value=0).fillna(0)
            x_no = chunk.reindex(columns=no_features, fill_value=0).fillna(0)
            p_with = predict_booster(with_model, x_with)
            p_no = predict_booster(no_model, x_no)

            delta = p_with - p_no
            margin = np.abs(p_no - threshold)
            pred_with = (p_with >= threshold).astype(np.int8)
            pred_no = (p_no >= threshold).astype(np.int8)
            good_direction = np.where(y_true == 1, delta > 0, delta < 0)

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
                    "no_utility_margin": margin,
                    "uncertainty_bin": bin_name_for(margin),
                    "delta": delta,
                    "abs_delta": np.abs(delta),
                    "good_direction": good_direction,
                    "class_flipped": pred_with != pred_no,
                    "with_correct": pred_with == y_true,
                    "no_correct": pred_no == y_true,
                    "logloss_improvement": binary_logloss(y_true, p_no)
                    - binary_logloss(y_true, p_with),
                }
            )
            yield out
            row_offset += n


def summarize_group(group: pd.DataFrame) -> Dict[str, object]:
    return {
        "rows": int(len(group)),
        "mean_abs_delta": float(group["abs_delta"].mean()),
        "p95_abs_delta": float(group["abs_delta"].quantile(0.95)),
        "mean_delta": float(group["delta"].mean()),
        "good_direction_rate": float(group["good_direction"].mean()),
        "mean_logloss_improvement": float(group["logloss_improvement"].mean()),
        "positive_logloss_improvement_rate": float((group["logloss_improvement"] > 0).mean()),
        "class_flip_rate": float(group["class_flipped"].mean()),
        "with_correct_rate": float(group["with_correct"].mean()),
        "no_correct_rate": float(group["no_correct"].mean()),
    }


def write_outputs(output_dir: Path, rows: pd.DataFrame, top_k: int, split: str, threshold: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, _, _ in UNCERTAINTY_BINS:
        group = rows[rows["uncertainty_bin"] == name]
        if group.empty:
            continue
        summary_rows.append({"uncertainty_bin": name, **summarize_group(group)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "uncertainty_delta_summary.csv", index=False)

    uncertain = rows[rows["no_utility_margin"] <= 0.05].copy()
    top_uncertain = uncertain.sort_values("abs_delta", ascending=False).head(top_k)
    top_uncertain.to_csv(output_dir / "top_uncertain_delta_rows.csv", index=False)

    payload = {
        "split": split,
        "threshold": threshold,
        "rows": int(len(rows)),
        "uncertain_rows_margin_le_0_05": int(len(uncertain)),
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "uncertainty_delta_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Utility Delta By No-Utility Uncertainty",
        "",
        f"- split: `{split}`",
        f"- threshold: `{threshold}`",
        f"- rows: `{len(rows)}`",
        f"- no-utility bizonytalan sorok, margin <= 0.05: `{len(uncertain)}`",
        "",
        "| no-utility margin bin | rows | mean abs delta | p95 abs delta | good direction | mean logloss improvement | class flip rate | with acc | no acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['uncertainty_bin']}` | {int(row['rows'])} | "
            f"{row['mean_abs_delta']:.6f} | {row['p95_abs_delta']:.6f} | "
            f"{row['good_direction_rate']:.4f} | {row['mean_logloss_improvement']:.6f} | "
            f"{row['class_flip_rate']:.4f} | {row['with_correct_rate']:.4f} | {row['no_correct_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Ertelmezes",
            "",
            "A `no_utility_margin` azt meri, hogy a no-utility modell probabilityje milyen messze van a thresholdtol.",
            "Minel kisebb ez az ertek, annal bizonytalanabb a no-utility modell class dontese.",
            "",
            "Ha a utility feature-ok foleg a bizonytalan sorokban mozditanak nagyobbat, akkor ez jo erv arra,",
            "hogy a utility nem globalis accuracy-javulaskent, hanem hatarhelyzeti plusz informaciokent hasznos.",
        ]
    )
    (output_dir / "uncertainty_delta_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    with_run_dir = Path(args.with_run_dir)
    no_run_dir = Path(args.no_run_dir)

    with_metrics = load_metrics(with_run_dir)
    no_metrics = load_metrics(no_run_dir)
    with_features = with_metrics["features"]
    no_features = no_metrics["features"]

    with_model = xgb.Booster()
    with_model.load_model(str(model_path_for(with_run_dir)))
    no_model = xgb.Booster()
    no_model.load_model(str(model_path_for(no_run_dir)))

    csv_items = manifest_paths(with_run_dir, args.split, args.max_csvs)
    print(f"Split: {args.split}, CSV count: {len(csv_items)}")

    chunks = list(
        stream_rows(
            csv_items=csv_items,
            with_features=with_features,
            no_features=no_features,
            with_model=with_model,
            no_model=no_model,
            threshold=args.threshold,
            chunk_size=args.chunk_size,
        )
    )
    rows = pd.concat(chunks, ignore_index=True)
    write_outputs(
        output_dir=Path(args.output_dir),
        rows=rows,
        top_k=args.top_k,
        split=args.split,
        threshold=args.threshold,
    )
    print(f"Mentve: {args.output_dir}")


if __name__ == "__main__":
    main()
