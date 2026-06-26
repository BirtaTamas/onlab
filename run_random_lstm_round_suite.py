from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random round suite: LSTM local explainability, XGBoost comparison, plot, utility-state analysis."
    )
    parser.add_argument(
        "--lstm-run-dir",
        type=Path,
        default=Path("artifacts/modellfutasok/lstm_sequence_full_cuda_final"),
    )
    parser.add_argument(
        "--xgboost-run-dir",
        type=Path,
        default=Path("artifacts/modellfutasok/xgboost_streaming_final_with_utility_100pct"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("processed_full"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/modellfutasok/probalstm"))
    parser.add_argument("--round-count", type=int, default=10)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_manifest_csv_path(csv_path: str) -> Path:
    normalized = str(csv_path).replace("\\", "/")
    return Path(normalized)


def pick_rounds(manifest_path: Path, count: int, min_rows: int, seed: int) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    test_manifest = manifest[manifest["split"] == "test"].copy()
    if test_manifest.empty:
        raise ValueError(f"Nincs test split a manifestben: {manifest_path}")

    rng = np.random.default_rng(seed)
    csv_paths = test_manifest["csv_path"].drop_duplicates().to_numpy()
    rng.shuffle(csv_paths)

    candidates = []
    for csv_path in csv_paths:
        csv_path_obj = normalize_manifest_csv_path(str(csv_path))
        try:
            df = pd.read_csv(csv_path_obj, usecols=["round_num"])
        except Exception as exc:
            print(f"CSV kihagyva ({csv_path_obj}): {exc}", flush=True)
            continue
        round_counts = df["round_num"].value_counts().reset_index()
        round_counts.columns = ["round_num", "rows"]
        round_counts = round_counts[round_counts["rows"] >= min_rows]
        for _, row in round_counts.iterrows():
            candidates.append(
                {
                    "csv_path": str(csv_path_obj),
                    "round_num": int(row["round_num"]),
                    "rows": int(row["rows"]),
                }
            )

    if len(candidates) < count:
        raise ValueError(f"Csak {len(candidates)} alkalmas roundot talaltam, kert: {count}")

    candidate_df = pd.DataFrame(candidates).drop_duplicates(["csv_path", "round_num"])
    chosen_idx = rng.choice(len(candidate_df), size=count, replace=False)
    return candidate_df.iloc[chosen_idx].reset_index(drop=True)


def fmt_float(value: object, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def summarize_utility_states(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for idx, path in enumerate(paths, start=1):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["suite_round_index"] = idx
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def cohort_summary(df: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    group = df[mask].copy()
    if group.empty:
        return {
            "rows": 0,
            "rounds": 0,
            "lstm_mean_probability": None,
            "xgboost_mean_probability": None,
            "lstm_closer": 0,
            "xgboost_closer": 0,
            "lstm_accuracy_at_0_5": None,
            "xgboost_accuracy_at_0_5": None,
        }
    y = group["ct_win"].astype(int)
    lstm_prob = group["lstm_probability"].astype(float)
    xgb_prob = group["xgboost_probability"].astype(float)
    return {
        "rows": int(len(group)),
        "rounds": int(group["suite_round_index"].nunique()),
        "lstm_mean_probability": float(lstm_prob.mean()),
        "xgboost_mean_probability": float(xgb_prob.mean()),
        "lstm_closer": int((group["closer_model"] == "lstm").sum()),
        "xgboost_closer": int((group["closer_model"] == "xgboost").sum()),
        "lstm_accuracy_at_0_5": float(((lstm_prob >= 0.5).astype(int) == y).mean()),
        "xgboost_accuracy_at_0_5": float(((xgb_prob >= 0.5).astype(int) == y).mean()),
    }


def canonical_feature_family(base_feature: str) -> str:
    text = str(base_feature)
    if "__" in text:
        text = text.split("__", 1)[1]

    if text.endswith("__flash_duration"):
        return "flash_duration"
    if text.endswith("__duck_amount"):
        return "duck_amount"
    if text.endswith("__utility_total"):
        return "utility_total"
    if text.endswith("__flash"):
        return "flash_inventory"
    if text.endswith("__smoke"):
        return "smoke_inventory"
    if text.endswith("__he"):
        return "he_inventory"
    if text.endswith("__molly"):
        return "molly_inventory"

    prefixes = ("T_", "CT_")
    for prefix in prefixes:
        if text.startswith(prefix):
            rest = text[len(prefix):]
            shared_names = {
                "flash_duration_sum",
                "flash_alpha_mean",
                "flashed_players",
                "smokes_last_5s",
                "flashes_last_5s",
                "he_last_5s",
                "mollies_last_5s",
                "utility_damage_last_5s",
                "active_smokes",
                "active_infernos",
                "utility_inv",
                "smoke_inv",
                "flash_inv",
                "he_inv",
                "molly_inv",
                "kills_last_3s",
                "damage_last_5s",
            }
            if rest in shared_names:
                return rest

    return text


def summarize_ridge_features(coef_paths: list[Path], top_n: int = 10) -> pd.DataFrame:
    rows = []
    for idx, path in enumerate(coef_paths, start=1):
        if not path.exists():
            continue
        df = pd.read_csv(path).head(top_n)
        for _, row in df.iterrows():
            rows.append(
                {
                    "suite_round_index": idx,
                    "feature": row["feature"],
                    "base_feature": row.get("base_feature", str(row["feature"]).split("__", 1)[-1]),
                    "feature_family": canonical_feature_family(
                        row.get("base_feature", str(row["feature"]).split("__", 1)[-1])
                    ),
                    "is_utility": bool(row.get("is_utility", False)),
                    "abs_coefficient": float(row["abs_coefficient"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    feature_df = pd.DataFrame(rows)
    unique_presence = feature_df.drop_duplicates(["suite_round_index", "feature_family", "is_utility"]).copy()
    return (
        unique_presence.groupby(["feature_family", "is_utility"], as_index=False)
        .agg(
            round_presence_count=("feature_family", "size"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
        )
        .sort_values(["round_presence_count", "mean_abs_coefficient"], ascending=False)
    )


def write_final_report(
    output_dir: Path,
    selected_rounds: pd.DataFrame,
    comparison_rows: list[dict],
    utility_df: pd.DataFrame,
    ridge_feature_summary: pd.DataFrame,
) -> None:
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "suite_comparison_summary.csv", index=False)
    if not utility_df.empty:
        utility_df.to_csv(output_dir / "suite_utility_states_all.csv", index=False)
    if not ridge_feature_summary.empty:
        ridge_feature_summary.to_csv(output_dir / "suite_ridge_top_features.csv", index=False)

    lstm_round_wins = int((comparison_df["winner_by_mean_abs_error"] == "lstm").sum())
    xgb_round_wins = int((comparison_df["winner_by_mean_abs_error"] == "xgboost").sum())
    total_lstm_closer = int(comparison_df["lstm_closer"].sum())
    total_xgb_closer = int(comparison_df["xgboost_closer"].sum())

    lines = [
        "# Random LSTM Round Suite",
        "",
        f"- rounds: `{len(comparison_df)}`",
        f"- LSTM round wins by MAE: `{lstm_round_wins}`",
        f"- XGBoost round wins by MAE: `{xgb_round_wins}`",
        f"- LSTM closer ticks total: `{total_lstm_closer}`",
        f"- XGBoost closer ticks total: `{total_xgb_closer}`",
        "",
        "## Selected Rounds",
        "",
        "| idx | rows | round_num | csv |",
        "|---:|---:|---:|---|",
    ]
    for idx, row in selected_rounds.iterrows():
        lines.append(f"| {idx + 1} | {int(row['rows'])} | {int(row['round_num'])} | `{row['csv_path']}` |")

    lines.extend(
        [
            "",
            "## Model Comparison",
            "",
            "| idx | true_ct_win | rows | winner | lstm_mae | xgb_mae | lstm_logloss | xgb_logloss | lstm_closer | xgb_closer |",
            "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in comparison_df.iterrows():
        lines.append(
            f"| {int(row['idx'])} | {int(row['true_ct_win'])} | {int(row['rows'])} | "
            f"{row['winner_by_mean_abs_error']} | {row['lstm_mean_abs_error']:.6f} | "
            f"{row['xgboost_mean_abs_error']:.6f} | {row['lstm_logloss']:.6f} | "
            f"{row['xgboost_logloss']:.6f} | {int(row['lstm_closer'])} | {int(row['xgboost_closer'])} |"
        )

    lines.extend(
        [
            "",
            "## Utility Cohorts Across Random Rounds",
            "",
            "| cohort | rows | rounds | lstm_mean_prob | xgb_mean_prob | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if utility_df.empty:
        lines.append("| no utility state data | 0 | 0 | NA | NA | 0 | 0 | NA | NA |")
    else:
        cohort_masks = {
            "all ticks": pd.Series(True, index=utility_df.index),
            "active/recent utility": utility_df["active_recent_utility"].astype(bool),
            "strong utility action": utility_df["strong_utility_action"].astype(bool),
            "utility damage": utility_df["utility_damage"].astype(bool),
            "active smoke/inferno": utility_df["active_smoke_inferno"].astype(bool),
            "recent utility last 5s": utility_df["recent_utility_last_5s"].astype(bool),
            "flash effect present": utility_df["flash_effect_present"].astype(bool),
        }
        for name, mask in cohort_masks.items():
            row = cohort_summary(utility_df, mask)
            lines.append(
                f"| {name} | {row['rows']} | {row['rounds']} | "
                f"{fmt_float(row['lstm_mean_probability'])} | {fmt_float(row['xgboost_mean_probability'])} | "
                f"{row['lstm_closer']} | {row['xgboost_closer']} | "
                f"{fmt_float(row['lstm_accuracy_at_0_5'])} | {fmt_float(row['xgboost_accuracy_at_0_5'])} |"
            )

    lines.extend(["", "## Frequent Top Ridge Features", ""])
    if ridge_feature_summary.empty:
        lines.append("- No ridge feature summary was produced.")
    else:
        lines.extend(["| feature csalad | utility | round_presence | mean_abs_coef | max_abs_coef |", "|---|---:|---:|---:|---:|"])
        for _, row in ridge_feature_summary.head(20).iterrows():
            lines.append(
                f"| `{row['feature_family']}` | {bool(row['is_utility'])} | {int(row['round_presence_count'])} | "
                f"{float(row['mean_abs_coefficient']):.6f} | {float(row['max_abs_coefficient']):.6f} |"
            )

    if not comparison_df.empty:
        if lstm_round_wins > xgb_round_wins:
            winner_text = "A random round mintan az LSTM tobb roundban volt jobb MAE szerint."
        elif xgb_round_wins > lstm_round_wins:
            winner_text = "A random round mintan az XGBoost tobb roundban volt jobb MAE szerint."
        else:
            winner_text = "A random round mintan az LSTM es az XGBoost round-szinten dontetlenul allt MAE szerint."
    else:
        winner_text = "Nem keszult osszehasonlithato round summary."

    lines.extend(
        [
            "",
            "## Final Conclusion Draft",
            "",
            winner_text,
            "",
            (
                "A lokalis ridge surrogate-ok celja nem uj prediktiv modell tanitasa, hanem az LSTM "
                "roundon beluli valoszinuseg-mozgasanak ertelmezheto kozelitese. A suite riportban ezert "
                "kulon erdemes kezelni a prediktiv osszehasonlitast es az explainability eredmenyeket."
            ),
            "",
            (
                "A utility cohort tabla azt mutatja, hogy az aktiv smoke/inferno, utility damage es recent utility "
                "helyzetekben melyik modell valoszinusege volt kozelebb a valos roundkimenethez. Ez lokalis parja "
                "a korabbi XGBoost utility ablation elemzesnek, de itt roundon beluli tick-szintu viselkedest mer."
            ),
        ]
    )

    (output_dir / "final_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_rounds = pick_rounds(
        manifest_path=args.lstm_run_dir / "sampled_split_manifest.csv",
        count=args.round_count,
        min_rows=args.min_rows,
        seed=args.random_seed,
    )
    selected_rounds.to_csv(args.output_dir / "selected_rounds.csv", index=False)

    comparison_rows = []
    utility_state_paths = []
    ridge_coef_paths = []

    for idx, row in selected_rounds.iterrows():
        round_idx = idx + 1
        csv_path = str(row["csv_path"])
        round_num = int(row["round_num"])
        round_tag = f"round_{round_idx:02d}"
        lstm_out = args.output_dir / round_tag / "lstm"
        compare_out = args.output_dir / round_tag / "compare"
        plot_path = compare_out / "local_round_probability_plot.png"

        if not args.skip_existing or not (compare_out / "local_round_lstm_xgboost_summary.json").exists():
            run_command(
                [
                    sys.executable,
                    "predict_lstm_local_round.py",
                    "--lstm-run-dir",
                    str(args.lstm_run_dir),
                    "--data-root",
                    str(args.data_root),
                    "--csv",
                    csv_path,
                    "--round-num",
                    str(round_num),
                    "--output-dir",
                    str(lstm_out),
                    "--device",
                    args.device,
                ]
            )
            run_command(
                [
                    sys.executable,
                    "compare_lstm_xgboost_round.py",
                    "--lstm-run-dir",
                    str(lstm_out),
                    "--xgboost-run-dir",
                    str(args.xgboost_run_dir),
                    "--output-dir",
                    str(compare_out),
                ]
            )
            run_command(
                [
                    sys.executable,
                    "plot_local_round_probabilities.py",
                    "--input-csv",
                    str(compare_out / "local_round_lstm_xgboost_predictions.csv"),
                    "--output-png",
                    str(plot_path),
                    "--show-ridge",
                ]
            )
            run_command(
                [
                    sys.executable,
                    "analyze_local_round_utility_states.py",
                    "--comparison-csv",
                    str(compare_out / "local_round_lstm_xgboost_predictions.csv"),
                    "--output-dir",
                    str(compare_out),
                ]
            )

        summary = load_json(compare_out / "local_round_lstm_xgboost_summary.json")
        comparison_rows.append(
            {
                "idx": round_idx,
                "csv_path": csv_path,
                "round_num": round_num,
                "rows": int(summary["row_count"]),
                "true_ct_win": int(pd.read_csv(compare_out / "local_round_lstm_xgboost_predictions.csv", nrows=1)["ct_win"].iloc[0]),
                "winner_by_mean_abs_error": summary["winner_by_mean_abs_error"],
                "winner_by_brier": summary["winner_by_brier"],
                "winner_by_logloss": summary["winner_by_logloss"],
                "lstm_mean_abs_error": float(summary["lstm"]["mean_abs_error"]),
                "xgboost_mean_abs_error": float(summary["xgboost"]["mean_abs_error"]),
                "lstm_brier": float(summary["lstm"]["brier_score"]),
                "xgboost_brier": float(summary["xgboost"]["brier_score"]),
                "lstm_logloss": float(summary["lstm"]["logloss"]),
                "xgboost_logloss": float(summary["xgboost"]["logloss"]),
                "lstm_accuracy_at_0_5": float(summary["lstm"]["accuracy_at_0_5"]),
                "xgboost_accuracy_at_0_5": float(summary["xgboost"]["accuracy_at_0_5"]),
                "lstm_closer": int(summary["closer_counts"]["lstm"]),
                "xgboost_closer": int(summary["closer_counts"]["xgboost"]),
                "tie": int(summary["closer_counts"]["tie"]),
            }
        )
        utility_state_paths.append(compare_out / "local_round_utility_states.csv")
        ridge_coef_paths.append(lstm_out / "local_round_ridge_coefficients.csv")

    utility_df = summarize_utility_states(utility_state_paths)
    ridge_feature_summary = summarize_ridge_features(ridge_coef_paths)
    write_final_report(
        output_dir=args.output_dir,
        selected_rounds=selected_rounds,
        comparison_rows=comparison_rows,
        utility_df=utility_df,
        ridge_feature_summary=ridge_feature_summary,
    )
    run_command(
        [
            sys.executable,
            "plot_lstm_feature_importance_report.py",
            "--suite-dir",
            str(args.output_dir),
        ]
    )
    print(f"Suite kesz: {args.output_dir / 'final_conclusion.md'}")


if __name__ == "__main__":
    main()
