import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from train_xgboost import is_utility_column


LABEL_COL = "ct_win"
ID_COLUMNS = ("round_num", "tick", "seconds_in_round")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kilistazza azokat a snapshotokat, ahol a with-utility es no-utility modell mast prediktal."
    )
    parser.add_argument("--with-run-dir", type=str, required=True)
    parser.add_argument("--no-run-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument(
        "--utility-columns",
        choices=["all", "compact"],
        default="all",
        help="all: minden utility feature; compact: csak osszesitett/ertelmezhetobb utility oszlopok.",
    )
    parser.add_argument(
        "--max-csvs",
        type=int,
        default=None,
        help="Opcionális gyors proba limit.",
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
    raise FileNotFoundError(f"Nem talaltam modell fajlt itt: {run_dir}")


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


def read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readline().strip().split(",")


def compact_utility_column(name: str) -> bool:
    lowered = name.lower()
    compact_tokens = (
        "utility_inv",
        "utility_damage",
        "active_smokes",
        "active_infernos",
        "smokes_last_5s",
        "flashes_last_5s",
        "he_last_5s",
        "mollies_last_5s",
        "flash_duration",
        "flash_inv",
        "smoke_inv",
        "molly_inv",
        "he_inv",
        "flash_inv_diff",
        "smoke_inv_diff",
        "molly_inv_diff",
        "utility_inv_diff",
    )
    return any(token in lowered for token in compact_tokens)


def utility_columns(header: Iterable[str], mode: str) -> List[str]:
    cols = []
    for col in header:
        if col == LABEL_COL or col in ID_COLUMNS:
            continue
        if not is_utility_column(col):
            continue
        if mode == "compact" and not compact_utility_column(col):
            continue
        cols.append(col)
    return cols


def predict_booster(booster: xgb.Booster, x_data: pd.DataFrame) -> np.ndarray:
    best_iteration = getattr(booster, "best_iteration", None)
    dmat = xgb.DMatrix(x_data)
    if best_iteration is not None and best_iteration >= 0:
        return booster.predict(dmat, iteration_range=(0, int(best_iteration) + 1))
    return booster.predict(dmat)


def stream_disagreements(
    csv_items: List[Tuple[str, Path]],
    with_features: List[str],
    no_features: List[str],
    with_model: xgb.Booster,
    no_model: xgb.Booster,
    threshold: float,
    chunk_size: int,
    utility_mode: str,
) -> Iterator[pd.DataFrame]:
    needed_features = set(with_features) | set(no_features) | {LABEL_COL} | set(ID_COLUMNS)

    for rel_csv, csv_path in csv_items:
        header = read_header(csv_path)
        util_cols = utility_columns(header, utility_mode)
        usecols = [col for col in header if col in needed_features or col in util_cols]

        row_offset = 0
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
            n = len(chunk)
            y_true = chunk[LABEL_COL].fillna(0).astype(np.int64).to_numpy()

            x_with = chunk.reindex(columns=with_features, fill_value=0).fillna(0)
            x_no = chunk.reindex(columns=no_features, fill_value=0).fillna(0)
            p_with = predict_booster(with_model, x_with)
            p_no = predict_booster(no_model, x_no)

            pred_with = (p_with >= threshold).astype(np.int8)
            pred_no = (p_no >= threshold).astype(np.int8)
            mask = pred_with != pred_no
            if not mask.any():
                row_offset += n
                continue

            disagreement = pd.DataFrame(
                {
                    "csv_path": rel_csv,
                    "csv_row_index": np.arange(row_offset, row_offset + n)[mask],
                    "round_num": chunk["round_num"].to_numpy()[mask] if "round_num" in chunk else np.nan,
                    "tick": chunk["tick"].to_numpy()[mask] if "tick" in chunk else np.nan,
                    "seconds_in_round": chunk["seconds_in_round"].to_numpy()[mask]
                    if "seconds_in_round" in chunk
                    else np.nan,
                    "ct_win": y_true[mask],
                    "p_with_utility": p_with[mask],
                    "p_no_utility": p_no[mask],
                    "pred_with_utility": pred_with[mask],
                    "pred_no_utility": pred_no[mask],
                    "delta": (p_with - p_no)[mask],
                    "abs_delta": np.abs(p_with - p_no)[mask],
                    "with_correct": pred_with[mask] == y_true[mask],
                    "no_correct": pred_no[mask] == y_true[mask],
                }
            )
            disagreement["winner_model"] = np.where(
                disagreement["with_correct"] & ~disagreement["no_correct"],
                "with_utility",
                np.where(
                    disagreement["no_correct"] & ~disagreement["with_correct"],
                    "no_utility",
                    "both_wrong",
                ),
            )

            utility_part = chunk.loc[mask, util_cols].reset_index(drop=True)
            yield pd.concat([disagreement.reset_index(drop=True), utility_part], axis=1)
            row_offset += n


def write_outputs(
    output_dir: Path,
    split: str,
    threshold: float,
    utility_mode: str,
    chunks: Iterator[pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "prediction_disagreements.csv"
    summary_path = output_dir / "prediction_disagreements_summary.json"
    md_path = output_dir / "prediction_disagreements_summary.md"

    total_rows = 0
    with_wins = 0
    no_wins = 0
    both_wrong = 0
    abs_delta_sum = 0.0
    first = True

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        for chunk in chunks:
            if chunk.empty:
                continue
            chunk.to_csv(handle, index=False, header=first)
            first = False

            total_rows += len(chunk)
            winners = chunk["winner_model"].value_counts()
            with_wins += int(winners.get("with_utility", 0))
            no_wins += int(winners.get("no_utility", 0))
            both_wrong += int(winners.get("both_wrong", 0))
            abs_delta_sum += float(chunk["abs_delta"].sum())

    summary = {
        "split": split,
        "threshold": threshold,
        "utility_columns": utility_mode,
        "disagreement_rows": total_rows,
        "with_utility_correct_no_utility_wrong": with_wins,
        "no_utility_correct_with_utility_wrong": no_wins,
        "both_wrong": both_wrong,
        "mean_abs_delta_on_disagreements": abs_delta_sum / total_rows if total_rows else None,
        "output_csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Prediction Disagreements",
        "",
        f"- split: `{split}`",
        f"- threshold: `{threshold}`",
        f"- utility columns: `{utility_mode}`",
        f"- disagreement rows: `{total_rows}`",
        f"- with utility correct, no utility wrong: `{with_wins}`",
        f"- no utility correct, with utility wrong: `{no_wins}`",
        f"- both wrong: `{both_wrong}`",
    ]
    if total_rows:
        lines.append(f"- mean abs delta on disagreements: `{summary['mean_abs_delta_on_disagreements']:.6f}`")
    lines.append(f"- csv: `{csv_path.name}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    print(f"With features: {len(with_features)}, no features: {len(no_features)}")

    chunks = stream_disagreements(
        csv_items=csv_items,
        with_features=with_features,
        no_features=no_features,
        with_model=with_model,
        no_model=no_model,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        utility_mode=args.utility_columns,
    )
    write_outputs(
        output_dir=Path(args.output_dir),
        split=args.split,
        threshold=args.threshold,
        utility_mode=args.utility_columns,
        chunks=chunks,
    )
    print(f"Keszen van: {args.output_dir}")


if __name__ == "__main__":
    main()
