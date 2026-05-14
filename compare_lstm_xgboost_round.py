import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


LABEL_COL = "ct_win"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ugyanazon lokalis roundon LSTM es XGBoost predikciok osszehasonlitasa."
    )
    parser.add_argument(
        "--lstm-run-dir",
        type=str,
        default="artifacts/modellfutasok/lstm_sequence_full_cuda_final",
    )
    parser.add_argument(
        "--xgboost-run-dir",
        type=str,
        default="artifacts/modellfutasok/xgboost_streaming_final_with_utility_100pct",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/modellfutasok/lstm_vs_xgboost_local_round",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readline().strip().split(",")


def clipped_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    probs = np.clip(y_prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    y = y_true.astype(np.float64)
    return float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    abs_error = np.abs(y_prob - y_true)
    return {
        "row_count": int(len(y_true)),
        "positive_count": int((y_true == 1).sum()),
        "negative_count": int((y_true == 0).sum()),
        "mean_probability": float(np.mean(y_prob)),
        "first_probability": float(y_prob[0]),
        "last_probability": float(y_prob[-1]),
        "min_probability": float(np.min(y_prob)),
        "max_probability": float(np.max(y_prob)),
        "accuracy_at_0_5": float(np.mean(y_pred == y_true)),
        "brier_score": float(np.mean(np.square(y_prob - y_true))),
        "mean_abs_error": float(np.mean(abs_error)),
        "median_abs_error": float(np.median(abs_error)),
        "logloss": clipped_logloss(y_true, y_prob),
    }


def predict_xgboost_round(
    xgboost_run_dir: Path,
    csv_path: Path,
    round_num: int,
    ticks: List[int],
) -> pd.DataFrame:
    metrics = load_json(xgboost_run_dir / "metrics.json")
    feature_columns = list(metrics["features"])
    header = read_header(csv_path)
    needed = set(feature_columns + [LABEL_COL, "round_num", "tick"])
    usecols = [col for col in header if col in needed]

    df = pd.read_csv(csv_path, usecols=usecols)
    if "round_num" not in df.columns or "tick" not in df.columns:
        raise ValueError("A round osszehasonlitashoz kell round_num es tick oszlop.")
    round_df = df[(df["round_num"] == round_num) & (df["tick"].isin(ticks))].copy()
    if round_df.empty:
        raise ValueError(f"Nem talaltam sorokat: csv={csv_path}, round_num={round_num}")

    round_df["tick"] = round_df["tick"].astype(int)
    round_df = round_df.set_index("tick").loc[ticks].reset_index()
    x_data = round_df.reindex(columns=feature_columns, fill_value=0).fillna(0)

    booster = xgb.Booster()
    booster.load_model(str(xgboost_run_dir / "xgboost_model.json"))
    dmat = xgb.DMatrix(x_data)

    best_iteration: Optional[int] = metrics.get("best_iteration")
    if isinstance(best_iteration, int) and best_iteration >= 0:
        probs = booster.predict(dmat, iteration_range=(0, best_iteration + 1))
    else:
        probs = booster.predict(dmat)

    out = round_df[["tick", "round_num", LABEL_COL]].copy()
    out["xgboost_probability"] = probs.astype(float)
    return out


def main() -> None:
    args = parse_args()
    lstm_run_dir = Path(args.lstm_run_dir)
    xgboost_run_dir = Path(args.xgboost_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lstm_round_path = lstm_run_dir / "local_round_predictions.csv"
    lstm_df = pd.read_csv(lstm_round_path)
    if lstm_df.empty:
        raise ValueError(f"Ures local round fajl: {lstm_round_path}")

    csv_path = Path(str(lstm_df["csv_path"].iloc[0]))
    round_num = int(lstm_df["round_num"].iloc[0])
    ticks = lstm_df["tick"].astype(int).tolist()

    xgb_df = predict_xgboost_round(
        xgboost_run_dir=xgboost_run_dir,
        csv_path=csv_path,
        round_num=round_num,
        ticks=ticks,
    )
    compare_df = lstm_df.merge(xgb_df, on=["tick", "round_num"], how="inner")
    if len(compare_df) != len(lstm_df):
        raise ValueError(
            f"Tick illesztes nem teljes: lstm={len(lstm_df)}, matched={len(compare_df)}"
        )

    y_true = compare_df[LABEL_COL].fillna(0).astype(np.int64).to_numpy()
    lstm_prob = compare_df["lstm_probability"].astype(float).to_numpy()
    xgb_prob = compare_df["xgboost_probability"].astype(float).to_numpy()

    compare_df["lstm_abs_error"] = np.abs(lstm_prob - y_true)
    compare_df["xgboost_abs_error"] = np.abs(xgb_prob - y_true)
    compare_df["closer_model"] = np.where(
        compare_df["lstm_abs_error"] < compare_df["xgboost_abs_error"],
        "lstm",
        np.where(
            compare_df["xgboost_abs_error"] < compare_df["lstm_abs_error"],
            "xgboost",
            "tie",
        ),
    )
    compare_df["probability_delta_lstm_minus_xgboost"] = lstm_prob - xgb_prob

    closer_counts = compare_df["closer_model"].value_counts().to_dict()
    summary = {
        "csv_path": str(csv_path),
        "round_num": round_num,
        "row_count": int(len(compare_df)),
        "lstm_run_dir": str(lstm_run_dir),
        "xgboost_run_dir": str(xgboost_run_dir),
        "lstm": binary_metrics(y_true, lstm_prob),
        "xgboost": binary_metrics(y_true, xgb_prob),
        "closer_counts": {
            "lstm": int(closer_counts.get("lstm", 0)),
            "xgboost": int(closer_counts.get("xgboost", 0)),
            "tie": int(closer_counts.get("tie", 0)),
        },
    }

    winner = min(
        ["lstm", "xgboost"],
        key=lambda name: summary[name]["mean_abs_error"],
    )
    summary["winner_by_mean_abs_error"] = winner
    summary["winner_by_brier"] = min(
        ["lstm", "xgboost"],
        key=lambda name: summary[name]["brier_score"],
    )
    summary["winner_by_logloss"] = min(
        ["lstm", "xgboost"],
        key=lambda name: summary[name]["logloss"],
    )

    compare_path = output_dir / "local_round_lstm_xgboost_predictions.csv"
    summary_path = output_dir / "local_round_lstm_xgboost_summary.json"
    report_path = output_dir / "local_round_lstm_xgboost_summary.md"

    compare_df.to_csv(compare_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# LSTM vs XGBoost Local Round Comparison",
        "",
        f"- csv_path: `{csv_path}`",
        f"- round_num: `{round_num}`",
        f"- rows: `{len(compare_df)}`",
        "",
        "## Metrics",
        "",
        "| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in ["lstm", "xgboost"]:
        row = summary[model_name]
        lines.append(
            f"| {model_name} | {row['mean_abs_error']:.6f} | {row['brier_score']:.6f} | "
            f"{row['logloss']:.6f} | {row['accuracy_at_0_5']:.6f} | {row['mean_probability']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Closer Per Tick",
            "",
            f"- lstm: `{summary['closer_counts']['lstm']}`",
            f"- xgboost: `{summary['closer_counts']['xgboost']}`",
            f"- tie: `{summary['closer_counts']['tie']}`",
            "",
            f"Winner by mean absolute error: `{summary['winner_by_mean_abs_error']}`",
            f"Winner by Brier score: `{summary['winner_by_brier']}`",
            f"Winner by logloss: `{summary['winner_by_logloss']}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Predikciok mentve: {compare_path}")
    print(f"Osszegzes mentve: {report_path}")


if __name__ == "__main__":
    main()
