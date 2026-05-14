from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Utility allapotok elemzese egy LSTM vs XGBoost lokalis round comparison fajlhoz."
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("artifacts/modellfutasok/lstm_vs_xgboost_random_tyloo_ra_r8/local_round_lstm_xgboost_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/modellfutasok/lstm_vs_xgboost_random_tyloo_ra_r8"),
    )
    return parser.parse_args()


def numeric_sum(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return df[available].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)


def build_state_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    smoke_total_cols = [
        "active_smokes_total",
    ]
    smoke_detection_cols = [
        "active_smokes_total",
        "T_active_smokes",
        "CT_active_smokes",
        "T_A_site_active_smokes",
        "CT_A_site_active_smokes",
        "T_B_site_active_smokes",
        "CT_B_site_active_smokes",
    ]
    inferno_total_cols = [
        "active_infernos_total",
    ]
    inferno_detection_cols = [
        "active_infernos_total",
        "T_active_infernos",
        "CT_active_infernos",
        "T_A_site_active_infernos",
        "CT_A_site_active_infernos",
        "T_B_site_active_infernos",
        "CT_B_site_active_infernos",
    ]
    utility_damage_cols = [
        "T_utility_damage_last_5s",
        "CT_utility_damage_last_5s",
        "utility_damage_diff_last_5s",
    ]
    recent_utility_cols = [
        "T_smokes_last_5s",
        "CT_smokes_last_5s",
        "T_flashes_last_5s",
        "CT_flashes_last_5s",
        "T_he_last_5s",
        "CT_he_last_5s",
        "T_A_site_smokes_last_5s",
        "CT_A_site_smokes_last_5s",
        "T_B_site_smokes_last_5s",
        "CT_B_site_smokes_last_5s",
        "T_A_site_flashes_last_5s",
        "CT_A_site_flashes_last_5s",
        "T_B_site_flashes_last_5s",
        "CT_B_site_flashes_last_5s",
        "T_A_site_he_last_5s",
        "CT_A_site_he_last_5s",
        "T_B_site_he_last_5s",
        "CT_B_site_he_last_5s",
    ]
    flash_effect_cols = [
        "T_flashed_players",
        "CT_flashed_players",
        "T_flash_duration_sum",
        "CT_flash_duration_sum",
        "T_flash_alpha_mean",
        "CT_flash_alpha_mean",
    ] + [f"{side}{idx}__flash_duration" for side in ["T", "CT"] for idx in range(1, 6)]

    out["active_smoke_inferno"] = (numeric_sum(df, smoke_detection_cols + inferno_detection_cols) > 0)
    out["utility_damage"] = (numeric_sum(df, utility_damage_cols).abs() > 0)
    out["recent_utility_last_5s"] = (numeric_sum(df, recent_utility_cols) > 0)
    out["flash_effect_present"] = (numeric_sum(df, flash_effect_cols) > 0)
    out["active_recent_utility"] = (
        out["active_smoke_inferno"] | out["utility_damage"] | out["recent_utility_last_5s"] | out["flash_effect_present"]
    )
    out["strong_utility_action"] = (
        out["active_smoke_inferno"] | out["utility_damage"] | out["recent_utility_last_5s"]
    )

    out["active_smoke_count"] = numeric_sum(df, smoke_total_cols)
    out["active_inferno_count"] = numeric_sum(df, inferno_total_cols)
    out["utility_damage_sum"] = numeric_sum(df, ["T_utility_damage_last_5s", "CT_utility_damage_last_5s"])
    out["recent_utility_count"] = numeric_sum(df, recent_utility_cols)
    out["flash_effect_sum"] = numeric_sum(df, flash_effect_cols)
    return out


def summarize_group(df: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    group = df[mask].copy()
    if group.empty:
        return {
            "rows": 0,
            "row_rate": 0.0,
            "lstm_mean_probability": None,
            "xgboost_mean_probability": None,
            "lstm_minus_xgboost_mean": None,
            "lstm_closer": 0,
            "xgboost_closer": 0,
            "tie": 0,
            "lstm_accuracy_at_0_5": None,
            "xgboost_accuracy_at_0_5": None,
        }

    y = group["ct_win"].astype(int)
    lstm_prob = group["lstm_probability"].astype(float)
    xgb_prob = group["xgboost_probability"].astype(float)
    closer_counts = group["closer_model"].value_counts().to_dict()
    return {
        "rows": int(len(group)),
        "row_rate": float(len(group) / len(df)),
        "lstm_mean_probability": float(lstm_prob.mean()),
        "xgboost_mean_probability": float(xgb_prob.mean()),
        "lstm_minus_xgboost_mean": float((lstm_prob - xgb_prob).mean()),
        "lstm_closer": int(closer_counts.get("lstm", 0)),
        "xgboost_closer": int(closer_counts.get("xgboost", 0)),
        "tie": int(closer_counts.get("tie", 0)),
        "lstm_accuracy_at_0_5": float(((lstm_prob >= 0.5).astype(int) == y).mean()),
        "xgboost_accuracy_at_0_5": float(((xgb_prob >= 0.5).astype(int) == y).mean()),
    }


def make_intervals(df: pd.DataFrame, flag_col: str) -> list[dict[str, float]]:
    intervals = []
    active = df[flag_col].astype(bool).to_numpy()
    if not active.any():
        return intervals
    seconds = df["seconds_in_round"].astype(float).to_numpy()
    start_idx = None
    for idx, value in enumerate(active):
        if value and start_idx is None:
            start_idx = idx
        if start_idx is not None and (not value or idx == len(active) - 1):
            end_idx = idx if value and idx == len(active) - 1 else idx - 1
            intervals.append(
                {
                    "start_seconds": float(seconds[start_idx]),
                    "end_seconds": float(seconds[end_idx]),
                    "rows": int(end_idx - start_idx + 1),
                }
            )
            start_idx = None
    return intervals


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(args.comparison_csv)
    if comparison.empty:
        raise ValueError(f"Ures comparison CSV: {args.comparison_csv}")

    csv_path = Path(str(comparison["csv_path"].iloc[0]))
    round_num = int(comparison["round_num"].iloc[0])
    ticks = comparison["tick"].astype(int).tolist()

    source = pd.read_csv(csv_path)
    source_round = source[(source["round_num"] == round_num) & (source["tick"].isin(ticks))].copy()
    source_round["tick"] = source_round["tick"].astype(int)
    source_round = source_round.set_index("tick").loc[ticks].reset_index()

    states = build_state_flags(source_round)
    merged = pd.concat([comparison.reset_index(drop=True), states.reset_index(drop=True)], axis=1)
    merged_path = args.output_dir / "local_round_utility_states.csv"
    merged.to_csv(merged_path, index=False)

    cohort_masks = {
        "all ticks": pd.Series(True, index=merged.index),
        "active/recent utility": merged["active_recent_utility"],
        "strong utility action": merged["strong_utility_action"],
        "utility damage": merged["utility_damage"],
        "active smoke/inferno": merged["active_smoke_inferno"],
        "recent utility last 5s": merged["recent_utility_last_5s"],
        "flash effect present": merged["flash_effect_present"],
    }
    summary = {name: summarize_group(merged, mask) for name, mask in cohort_masks.items()}

    lines = [
        "# Local Round Utility State Analysis",
        "",
        f"- csv_path: `{csv_path}`",
        f"- round_num: `{round_num}`",
        f"- rows: `{len(merged)}`",
        f"- true ct_win: `{int(merged['ct_win'].iloc[0])}`",
        "",
        "## Cohort Summary",
        "",
        "| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary.items():
        def fmt(value):
            return "NA" if value is None else f"{value:.6f}"

        lines.append(
            f"| {name} | {row['rows']} | {row['row_rate']:.3f} | "
            f"{fmt(row['lstm_mean_probability'])} | {fmt(row['xgboost_mean_probability'])} | "
            f"{fmt(row['lstm_minus_xgboost_mean'])} | {row['lstm_closer']} | {row['xgboost_closer']} | "
            f"{fmt(row['lstm_accuracy_at_0_5'])} | {fmt(row['xgboost_accuracy_at_0_5'])} |"
        )

    lines.extend(["", "## Active Smoke/Inferno Intervals", ""])
    intervals = make_intervals(merged, "active_smoke_inferno")
    if not intervals:
        lines.append("- No active smoke/inferno interval in this local round.")
    else:
        for interval in intervals:
            lines.append(
                f"- `{interval['start_seconds']:.1f}s` - `{interval['end_seconds']:.1f}s`, rows `{interval['rows']}`"
            )

    lines.extend(["", "## Biggest LSTM-XGBoost Differences During Utility States", ""])
    utility_rows = merged[merged["strong_utility_action"]].copy()
    if utility_rows.empty:
        lines.append("- No strong utility action rows.")
    else:
        utility_rows["abs_lstm_minus_xgb"] = (
            utility_rows["lstm_probability"].astype(float) - utility_rows["xgboost_probability"].astype(float)
        ).abs()
        for _, row in utility_rows.sort_values("abs_lstm_minus_xgb", ascending=False).head(10).iterrows():
            lines.append(
                f"- seconds `{float(row['seconds_in_round']):.1f}`, "
                f"LSTM `{float(row['lstm_probability']):.4f}`, "
                f"XGBoost `{float(row['xgboost_probability']):.4f}`, "
                f"closer `{row['closer_model']}`, "
                f"smoke `{float(row['active_smoke_count']):.0f}`, "
                f"inferno `{float(row['active_inferno_count']):.0f}`, "
                f"utility_damage `{float(row['utility_damage_sum']):.1f}`, "
                f"recent_utility `{float(row['recent_utility_count']):.0f}`"
            )

    report_path = args.output_dir / "local_round_utility_state_summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Utility state CSV mentve: {merged_path}")
    print(f"Utility state riport mentve: {report_path}")


if __name__ == "__main__":
    main()
