from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_series(csv_path: Path) -> dict[str, list[float]]:
    series = {
        "seconds": [],
        "lstm": [],
        "xgboost": [],
        "ridge": [],
        "truth": [],
        "bomb_planted": [],
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"seconds_in_round", "lstm_probability", "xgboost_probability", "ct_win"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Hianyzo oszlopok: {', '.join(sorted(missing))}")

        has_ridge = "ridge_probability" in (reader.fieldnames or [])
        has_bomb = "bomb_planted" in (reader.fieldnames or [])

        for row in reader:
            series["seconds"].append(float(row["seconds_in_round"]))
            series["lstm"].append(float(row["lstm_probability"]))
            series["xgboost"].append(float(row["xgboost_probability"]))
            series["truth"].append(float(row["ct_win"]))
            if has_ridge:
                series["ridge"].append(float(row["ridge_probability"]))
            if has_bomb:
                series["bomb_planted"].append(float(row["bomb_planted"]))

    return series


def plot_probabilities(input_csv: Path, output_png: Path, show_ridge: bool) -> None:
    data = read_series(input_csv)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=140)
    ax.plot(data["seconds"], data["lstm"], label="LSTM CT win probability", color="#2563eb", linewidth=2.0)
    ax.plot(data["seconds"], data["xgboost"], label="XGBoost CT win probability", color="#dc2626", linewidth=2.0)

    if show_ridge and data["ridge"]:
        ax.plot(
            data["seconds"],
            data["ridge"],
            label="Local ridge approximation of LSTM",
            color="#16a34a",
            linewidth=1.6,
            linestyle="--",
            alpha=0.9,
        )

    if data["truth"]:
        truth_value = data["truth"][0]
        ax.axhline(
            truth_value,
            color="#111827",
            linestyle=":",
            linewidth=1.6,
            label=f"Actual ct_win = {int(truth_value)}",
        )

    if data["bomb_planted"] and any(value >= 0.5 for value in data["bomb_planted"]):
        first_bomb_second = next(
            second for second, planted in zip(data["seconds"], data["bomb_planted"]) if planted >= 0.5
        )
        ax.axvline(
            first_bomb_second,
            color="#f59e0b",
            linestyle="-.",
            linewidth=1.4,
            label=f"Bomb planted: {first_bomb_second:.1f}s",
        )

    ax.set_title("Round-level CT win probability by tick")
    ax.set_xlabel("Seconds in round")
    ax.set_ylabel("Predicted CT win probability")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot LSTM and XGBoost tick-by-tick probabilities for the selected local round."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("artifacts/modellfutasok/lstm_vs_xgboost_local_round/local_round_lstm_xgboost_predictions.csv"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("artifacts/modellfutasok/lstm_vs_xgboost_local_round/local_round_probability_plot.png"),
    )
    parser.add_argument("--show-ridge", action="store_true", help="Also draw the local ridge approximation.")
    args = parser.parse_args()

    plot_probabilities(args.input_csv, args.output_png, args.show_ridge)
    print(f"Grafikon mentve ide: {args.output_png}")


if __name__ == "__main__":
    main()
