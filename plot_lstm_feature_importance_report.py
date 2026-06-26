from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


UTILITY_COLOR = "#16a34a"
NON_UTILITY_COLOR = "#2563eb"
HIGHLIGHT_COLOR = "#f59e0b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Beszamolohoz valo LSTM feature-importance abrak generalasa a random round suite outputjaibol."
    )
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=Path("artifacts/modellfutasok/probalstm"),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Hany feature szerepeljen a globalis abrakon.",
    )
    parser.add_argument(
        "--case-count",
        type=int,
        default=3,
        help="Hany kiemelt esetre keszuljon kulon abra.",
    )
    parser.add_argument(
        "--utility-scan-top-n",
        type=int,
        default=80,
        help="Hany lokalis top feature-bol epuljon a utility-kategoriak osszesitese.",
    )
    return parser.parse_args()


def shorten_csv_path(csv_path: str) -> str:
    normalized = str(csv_path).replace("\\", "/")
    parts = Path(normalized).parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return normalized


def clean_feature_name(name: str) -> str:
    text = str(name)
    if "__" in text:
        _, text = text.split("__", 1)
    return text.replace("_", " ")


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


def utility_category(base_feature: str) -> str | None:
    text = str(base_feature)
    lowered = text.lower()

    if "flash_duration" in lowered or "flash_alpha" in lowered or "flashed_players" in lowered:
        return "flash_hatas"
    if "utility_damage" in lowered:
        return "utility_damage"
    if "active_smokes" in lowered or "smokes_last_5s" in lowered:
        return "site_smoke" if "_a_site_" in lowered or "_b_site_" in lowered else "smoke"
    if "active_infernos" in lowered or "mollies_last_5s" in lowered:
        return "site_inferno" if "_a_site_" in lowered or "_b_site_" in lowered else "inferno"
    if "flashes_last_5s" in lowered or "he_last_5s" in lowered:
        return "site_recent_utility" if "_a_site_" in lowered or "_b_site_" in lowered else "recent_utility"
    if "utility_inv" in lowered or "utility_total" in lowered:
        return "utility_inventory"
    if "flash_inv" in lowered or lowered.endswith("__flash"):
        return "flash_inventory"
    if "smoke_inv" in lowered or lowered.endswith("__smoke"):
        return "smoke_inventory"
    if "he_inv" in lowered or lowered.endswith("__he"):
        return "he_inventory"
    if "molly_inv" in lowered or lowered.endswith("__molly"):
        return "molly_inventory"
    return None


def utility_category_label(category: str) -> str:
    labels = {
        "flash_hatas": "flash hatasok",
        "utility_damage": "utility damage",
        "site_smoke": "site smoke-ok",
        "smoke": "smoke hatasok",
        "site_inferno": "site infernok",
        "inferno": "inferno hatasok",
        "site_recent_utility": "site utility aktivitasa",
        "recent_utility": "friss utility aktivitasa",
        "utility_inventory": "osszes utility inventory",
        "flash_inventory": "flash inventory",
        "smoke_inventory": "smoke inventory",
        "he_inventory": "HE inventory",
        "molly_inventory": "molly inventory",
    }
    return labels.get(category, category.replace("_", " "))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def rebuild_grouped_summary_from_rounds(suite_dir: Path, top_n: int) -> pd.DataFrame:
    rows = []
    for idx in range(1, 1000):
        coef_path = suite_dir / f"round_{idx:02d}" / "lstm" / "local_round_ridge_coefficients.csv"
        if not coef_path.exists():
            if idx > 50:
                break
            continue
        df = pd.read_csv(coef_path).head(top_n).copy()
        if df.empty:
            continue
        df["suite_round_index"] = idx
        df["feature_family"] = df["base_feature"].map(canonical_feature_family)
        df = df.drop_duplicates(["suite_round_index", "feature_family", "is_utility"])
        for _, row in df.iterrows():
            rows.append(
                {
                    "suite_round_index": int(row["suite_round_index"]),
                    "feature_family": str(row["feature_family"]),
                    "is_utility": bool(row["is_utility"]),
                    "abs_coefficient": float(row["abs_coefficient"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    grouped = pd.DataFrame(rows)
    return (
        grouped.groupby(["feature_family", "is_utility"], as_index=False)
        .agg(
            round_presence_count=("feature_family", "size"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
        )
        .sort_values(["round_presence_count", "mean_abs_coefficient", "max_abs_coefficient"], ascending=False)
    )


def rebuild_utility_category_summary_from_rounds(suite_dir: Path, top_n: int) -> pd.DataFrame:
    rows = []
    for idx in range(1, 1000):
        coef_path = suite_dir / f"round_{idx:02d}" / "lstm" / "local_round_ridge_coefficients.csv"
        if not coef_path.exists():
            if idx > 50:
                break
            continue
        df = pd.read_csv(coef_path).head(top_n).copy()
        if df.empty:
            continue
        df = df[df["is_utility"] == True].copy()
        if df.empty:
            continue
        df["suite_round_index"] = idx
        df["utility_category"] = df["base_feature"].map(utility_category)
        df = df[df["utility_category"].notna()].copy()
        df = df.drop_duplicates(["suite_round_index", "utility_category"])
        for _, row in df.iterrows():
            rows.append(
                {
                    "suite_round_index": int(row["suite_round_index"]),
                    "utility_category": str(row["utility_category"]),
                    "abs_coefficient": float(row["abs_coefficient"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    grouped = pd.DataFrame(rows)
    return (
        grouped.groupby("utility_category", as_index=False)
        .agg(
            round_presence_count=("utility_category", "size"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
        )
        .sort_values(["round_presence_count", "mean_abs_coefficient", "max_abs_coefficient"], ascending=False)
    )


def plot_global_features(
    suite_dir: Path,
    suite_df: pd.DataFrame,
    top_n: int,
    utility_scan_top_n: int,
) -> list[Path]:
    outputs: list[Path] = []
    normalized = rebuild_grouped_summary_from_rounds(suite_dir, top_n=10)
    if normalized.empty:
        if suite_df.empty:
            return outputs
        normalized = suite_df.copy()
        if "feature_family" not in normalized.columns:
            base_col = "base_feature" if "base_feature" in normalized.columns else "feature"
            normalized["feature_family"] = normalized[base_col].map(canonical_feature_family)
        count_col = "top10_count"
    else:
        count_col = "round_presence_count"

    normalized.sort_values(
        [count_col, "mean_abs_coefficient", "max_abs_coefficient"],
        ascending=False,
    ).to_csv(suite_dir / "feature_importance_grouped_summary.csv", index=False)

    ranked = normalized.sort_values(
        [count_col, "mean_abs_coefficient", "max_abs_coefficient"],
        ascending=False,
    ).head(top_n)
    labels = [clean_feature_name(name) for name in ranked["feature_family"]]
    colors = [UTILITY_COLOR if flag else NON_UTILITY_COLOR for flag in ranked["is_utility"]]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    y_pos = np.arange(len(ranked))
    ax.barh(y_pos, ranked[count_col], color=colors, alpha=0.92)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Rounds where the feature family appears in the local top list")
    ax.set_title("LSTM local surrogate: recurring important feature families")
    ax.grid(axis="x", color="#d1d5db", linewidth=0.8, alpha=0.7)
    for idx, (_, row) in enumerate(ranked.iterrows()):
        ax.text(
            float(row[count_col]) + 0.05,
            idx,
            f"mean |coef| = {float(row['mean_abs_coefficient']):.4f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    output = suite_dir / "feature_importance_overall_top_features.png"
    ensure_parent(output)
    fig.savefig(output)
    plt.close(fig)
    outputs.append(output)

    utility_category_df = rebuild_utility_category_summary_from_rounds(
        suite_dir,
        top_n=utility_scan_top_n,
    )
    if not utility_category_df.empty:
        utility_category_df = utility_category_df.sort_values(
            ["round_presence_count", "mean_abs_coefficient", "max_abs_coefficient"],
            ascending=False,
        )
        utility_category_df.to_csv(suite_dir / "feature_importance_utility_category_summary.csv", index=False)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        plot_df = utility_category_df.copy()
        labels = [utility_category_label(name) for name in plot_df["utility_category"]]
        y_pos = np.arange(len(plot_df))
        ax.barh(y_pos, plot_df["round_presence_count"], color=UTILITY_COLOR, alpha=0.92)
        ax.set_yticks(y_pos, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Rounds where the utility type appears in the expanded local top list")
        ax.set_title("LSTM local surrogate: recurring utility types across rounds")
        ax.grid(axis="x", color="#d1d5db", linewidth=0.8, alpha=0.7)
        for idx, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(
                float(row["round_presence_count"]) + 0.05,
                idx,
                f"mean |coef| = {float(row['mean_abs_coefficient']):.4f}",
                va="center",
                fontsize=9,
            )
        fig.tight_layout()
        output = suite_dir / "feature_importance_utility_top_features.png"
        fig.savefig(output)
        plt.close(fig)
        outputs.append(output)
    else:
        utility_df = ranked[ranked["is_utility"]].copy()
        if utility_df.empty:
            utility_df = normalized[normalized["is_utility"]].sort_values(
                [count_col, "mean_abs_coefficient", "max_abs_coefficient"],
                ascending=False,
            ).head(top_n)
        if not utility_df.empty:
            labels = [clean_feature_name(name) for name in utility_df["feature_family"]]
            fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
            y_pos = np.arange(len(utility_df))
            ax.barh(y_pos, utility_df[count_col], color=UTILITY_COLOR, alpha=0.92)
            ax.set_yticks(y_pos, labels)
            ax.invert_yaxis()
            ax.set_xlabel("Rounds where the feature family appears in the local top list")
            ax.set_title("LSTM local surrogate: recurring utility-related feature families")
            ax.grid(axis="x", color="#d1d5db", linewidth=0.8, alpha=0.7)
            for idx, (_, row) in enumerate(utility_df.iterrows()):
                ax.text(
                    float(row[count_col]) + 0.05,
                    idx,
                    f"mean |coef| = {float(row['mean_abs_coefficient']):.4f}",
                    va="center",
                    fontsize=9,
                )
            fig.tight_layout()
            output = suite_dir / "feature_importance_utility_top_features.png"
            fig.savefig(output)
            plt.close(fig)
            outputs.append(output)

    return outputs


def pick_case_rounds(suite_dir: Path, case_count: int) -> pd.DataFrame:
    selected = pd.read_csv(suite_dir / "selected_rounds.csv")
    rows = []
    for idx, row in selected.iterrows():
        round_idx = idx + 1
        coef_path = suite_dir / f"round_{round_idx:02d}" / "lstm" / "local_round_ridge_coefficients.csv"
        utility_path = suite_dir / f"round_{round_idx:02d}" / "compare" / "local_round_utility_states.csv"
        if not coef_path.exists() or not utility_path.exists():
            continue
        coef_df = pd.read_csv(coef_path).head(15)
        utility_df = pd.read_csv(utility_path)
        top_utility = coef_df[coef_df["is_utility"] == True].copy()
        rows.append(
            {
                "idx": round_idx,
                "csv_path": str(row["csv_path"]),
                "round_num": int(row["round_num"]),
                "rows": int(row["rows"]),
                "top15_utility_count": int(len(top_utility)),
                "top_utility_abs_coefficient": (
                    float(top_utility["abs_coefficient"].max()) if not top_utility.empty else 0.0
                ),
                "strong_utility_rows": int(utility_df["strong_utility_action"].astype(bool).sum()),
                "max_model_delta": float(utility_df["probability_delta_lstm_minus_xgboost"].abs().max()),
            }
        )
    case_df = pd.DataFrame(rows)
    if case_df.empty:
        return case_df
    return case_df.sort_values(
        ["top15_utility_count", "top_utility_abs_coefficient", "strong_utility_rows", "max_model_delta"],
        ascending=False,
    ).head(case_count)


def choose_focus_tick(contrib_df: pd.DataFrame, utility_states: pd.DataFrame) -> int | None:
    if contrib_df.empty:
        return None
    merged = contrib_df.merge(
        utility_states[["tick", "strong_utility_action"]],
        on="tick",
        how="left",
    )
    merged["strong_utility_action"] = merged["strong_utility_action"].fillna(False).astype(bool)
    utility_rows = merged[merged["strong_utility_action"] & merged["is_utility"].astype(bool)].copy()
    if not utility_rows.empty:
        grouped = (
            utility_rows.groupby("tick", as_index=False)["ridge_delta_contribution"]
            .apply(lambda s: float(np.abs(s).sum()))
            .rename(columns={"ridge_delta_contribution": "utility_contribution_sum"})
            .sort_values("utility_contribution_sum", ascending=False)
        )
        if not grouped.empty:
            return int(grouped.iloc[0]["tick"])
    grouped_all = (
        merged.groupby("tick", as_index=False)["ridge_delta_contribution"]
        .apply(lambda s: float(np.abs(s).sum()))
        .rename(columns={"ridge_delta_contribution": "contribution_sum"})
        .sort_values("contribution_sum", ascending=False)
    )
    if grouped_all.empty:
        return None
    return int(grouped_all.iloc[0]["tick"])


def plot_case_figure(suite_dir: Path, case_row: pd.Series) -> Path | None:
    idx = int(case_row["idx"])
    round_dir = suite_dir / f"round_{idx:02d}"
    coef_path = round_dir / "lstm" / "local_round_ridge_coefficients.csv"
    contrib_path = round_dir / "lstm" / "local_round_jump_contributions.csv"
    utility_path = round_dir / "compare" / "local_round_utility_states.csv"
    if not coef_path.exists() or not contrib_path.exists() or not utility_path.exists():
        return None

    coef_df = pd.read_csv(coef_path).head(10).copy()
    contrib_df = pd.read_csv(contrib_path).copy()
    utility_states = pd.read_csv(utility_path).copy()
    focus_tick = choose_focus_tick(contrib_df, utility_states)
    if focus_tick is None:
        return None

    jump_df = contrib_df[contrib_df["tick"] == focus_tick].copy()
    if jump_df.empty:
        return None
    jump_df = jump_df.sort_values("ridge_delta_contribution", key=lambda s: s.abs(), ascending=False).head(8)

    state_row = utility_states[utility_states["tick"] == focus_tick].head(1)
    seconds_text = "NA"
    lstm_prob_text = "NA"
    xgb_prob_text = "NA"
    if not state_row.empty:
        seconds_text = f"{float(state_row.iloc[0]['seconds_in_round']):.1f}s"
        lstm_prob_text = f"{float(state_row.iloc[0]['lstm_probability']):.3f}"
        xgb_prob_text = f"{float(state_row.iloc[0]['xgboost_probability']):.3f}"

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=160)

    coef_colors = [UTILITY_COLOR if flag else NON_UTILITY_COLOR for flag in coef_df["is_utility"]]
    coef_labels = [clean_feature_name(name) for name in coef_df["feature"]]
    coef_values = coef_df["coefficient"].astype(float).to_numpy()
    coef_pos = np.arange(len(coef_df))
    axes[0].barh(coef_pos, coef_values, color=coef_colors, alpha=0.92)
    axes[0].set_yticks(coef_pos, coef_labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0.0, color="#111827", linewidth=1.0)
    axes[0].set_title("Top local ridge coefficients")
    axes[0].set_xlabel("Coefficient sign and magnitude")
    axes[0].grid(axis="x", color="#d1d5db", linewidth=0.8, alpha=0.7)

    jump_colors = [UTILITY_COLOR if flag else HIGHLIGHT_COLOR for flag in jump_df["is_utility"]]
    jump_labels = [clean_feature_name(name) for name in jump_df["feature"]]
    jump_values = jump_df["ridge_delta_contribution"].astype(float).to_numpy()
    jump_pos = np.arange(len(jump_df))
    axes[1].barh(jump_pos, jump_values, color=jump_colors, alpha=0.92)
    axes[1].set_yticks(jump_pos, jump_labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_title(f"Top contributors at tick {focus_tick}")
    axes[1].set_xlabel("Contribution to local probability jump")
    axes[1].grid(axis="x", color="#d1d5db", linewidth=0.8, alpha=0.7)

    fig.suptitle(
        " | ".join(
            [
                f"Case {idx:02d}",
                shorten_csv_path(str(case_row['csv_path'])),
                f"round {int(case_row['round_num'])}",
                f"focus tick {focus_tick} ({seconds_text})",
                f"LSTM {lstm_prob_text}",
                f"XGBoost {xgb_prob_text}",
            ]
        ),
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output = suite_dir / f"feature_importance_case_round_{idx:02d}.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def write_summary(suite_dir: Path, global_paths: list[Path], case_df: pd.DataFrame, case_paths: list[Path]) -> Path:
    lines = [
        "# LSTM Feature Importance Figure Notes",
        "",
        "## Global figures",
        "",
    ]
    if not global_paths:
        lines.append("- Nem keszult globalis feature-importance abra.")
    else:
        for path in global_paths:
            lines.append(f"- `{path.name}`")

    lines.extend(["", "## Suggested case-study figures", ""])
    if case_df.empty or not case_paths:
        lines.append("- Nem sikerult kiemelt esetabrakat generalni.")
    else:
        for _, row in case_df.iterrows():
            idx = int(row["idx"])
            lines.append(
                f"- case `{idx:02d}`: round `{int(row['round_num'])}`, "
                f"top15 utility feature count `{int(row['top15_utility_count'])}`, "
                f"strong utility rows `{int(row['strong_utility_rows'])}`, "
                f"max |LSTM-XGBoost| `{float(row['max_model_delta']):.3f}`"
            )

    output = suite_dir / "feature_importance_figure_notes.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir

    suite_feature_path = suite_dir / "suite_ridge_top_features.csv"
    if not suite_feature_path.exists():
        raise FileNotFoundError(f"Nem talalom a suite ridge summary-t: {suite_feature_path}")

    suite_df = pd.read_csv(suite_feature_path)
    global_paths = plot_global_features(
        suite_dir,
        suite_df,
        top_n=args.top_n,
        utility_scan_top_n=args.utility_scan_top_n,
    )

    case_df = pick_case_rounds(suite_dir, case_count=args.case_count)
    case_paths: list[Path] = []
    for _, case_row in case_df.iterrows():
        case_path = plot_case_figure(suite_dir, case_row)
        if case_path is not None:
            case_paths.append(case_path)

    summary_path = write_summary(suite_dir, global_paths, case_df, case_paths)

    print(f"Global abrak: {[path.name for path in global_paths]}")
    print(f"Esetabrak: {[path.name for path in case_paths]}")
    print(f"Jegyzet: {summary_path}")


if __name__ == "__main__":
    main()
