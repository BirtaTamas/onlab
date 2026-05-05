import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from train_xgboost import is_utility_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class flip elemzes utility-aktiv snapshotokra a prediction_disagreements.csv alapjan."
    )
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=200)
    return parser.parse_args()


def numeric_sum(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)


def numeric_abs_sum(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).abs().sum(axis=1)


def find_utility_columns(columns: Iterable[str]) -> Dict[str, List[str]]:
    utility_cols = [col for col in columns if is_utility_column(col)]
    return {
        "all": utility_cols,
        "inventory": [
            col
            for col in utility_cols
            if col.lower().endswith("_inv")
            or col.lower().endswith("__smoke")
            or col.lower().endswith("__flash")
            or col.lower().endswith("__he")
            or col.lower().endswith("__molly")
            or col.lower().endswith("__utility_total")
            or "utility_inv_diff" in col.lower()
            or "flash_inv_diff" in col.lower()
            or "smoke_inv_diff" in col.lower()
            or "molly_inv_diff" in col.lower()
        ],
        "active": [
            col
            for col in utility_cols
            if "active_smokes" in col.lower() or "active_infernos" in col.lower()
        ],
        "recent": [col for col in utility_cols if "last_5s" in col.lower()],
        "damage": [col for col in utility_cols if "utility_damage" in col.lower()],
        "flash_effect": [col for col in utility_cols if "flash_duration" in col.lower()],
        "smoke": [col for col in utility_cols if "smoke" in col.lower()],
        "flash": [col for col in utility_cols if "flash" in col.lower()],
        "he": [col for col in utility_cols if "_he" in col.lower() or col.lower().endswith("__he")],
        "molly_inferno": [
            col for col in utility_cols if "molly" in col.lower() or "inferno" in col.lower()
        ],
    }


def summarize_mask(df: pd.DataFrame, mask: pd.Series, name: str) -> Dict[str, object]:
    sub = df[mask].copy()
    if sub.empty:
        return {
            "group": name,
            "rows": 0,
        }
    with_wins = int((sub["winner_model"] == "with_utility").sum())
    no_wins = int((sub["winner_model"] == "no_utility").sum())
    return {
        "group": name,
        "rows": int(len(sub)),
        "with_utility_correct": with_wins,
        "no_utility_correct": no_wins,
        "with_minus_no": with_wins - no_wins,
        "with_win_rate": with_wins / len(sub),
        "mean_abs_delta": float(sub["abs_delta"].mean()),
        "mean_delta": float(sub["delta"].mean()),
        "ct_win_rate": float(pd.to_numeric(sub["ct_win"], errors="coerce").mean()),
    }


def add_activity_columns(df: pd.DataFrame, groups: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    out["utility_abs_sum_all"] = numeric_abs_sum(out, groups["all"])
    out["utility_inventory_abs_sum"] = numeric_abs_sum(out, groups["inventory"])
    out["utility_active_sum"] = numeric_abs_sum(out, groups["active"])
    out["utility_recent_sum"] = numeric_abs_sum(out, groups["recent"])
    out["utility_damage_sum"] = numeric_abs_sum(out, groups["damage"])
    out["utility_flash_effect_sum"] = numeric_abs_sum(out, groups["flash_effect"])
    out["utility_smoke_sum"] = numeric_abs_sum(out, groups["smoke"])
    out["utility_flash_sum"] = numeric_abs_sum(out, groups["flash"])
    out["utility_he_sum"] = numeric_abs_sum(out, groups["he"])
    out["utility_molly_inferno_sum"] = numeric_abs_sum(out, groups["molly_inferno"])

    out["has_any_utility"] = out["utility_abs_sum_all"] > 0
    out["has_active_or_recent_utility"] = (
        (out["utility_active_sum"] > 0)
        | (out["utility_recent_sum"] > 0)
        | (out["utility_damage_sum"] > 0)
        | (out["utility_flash_effect_sum"] > 0)
    )
    out["has_utility_damage"] = out["utility_damage_sum"] > 0
    out["has_active_smoke_or_inferno"] = out["utility_active_sum"] > 0
    out["has_recent_utility"] = out["utility_recent_sum"] > 0
    out["has_flash_effect"] = out["utility_flash_effect_sum"] > 0
    out["strong_utility_action"] = (
        (out["utility_active_sum"] >= 2)
        | (out["utility_recent_sum"] >= 2)
        | (out["utility_damage_sum"] >= 10)
        | (out["utility_flash_effect_sum"] >= 1)
    )
    return out


def write_summary(output_dir: Path, enriched: pd.DataFrame, summary: pd.DataFrame, top_k: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_dir / "utility_active_flip_rows.csv", index=False)
    summary.to_csv(output_dir / "utility_active_flip_summary.csv", index=False)

    top_good = (
        enriched[enriched["winner_model"] == "with_utility"]
        .sort_values("abs_delta", ascending=False)
        .head(top_k)
    )
    top_bad = (
        enriched[enriched["winner_model"] == "no_utility"]
        .sort_values("abs_delta", ascending=False)
        .head(top_k)
    )
    top_good.to_csv(output_dir / "top_utility_good_flips.csv", index=False)
    top_bad.to_csv(output_dir / "top_utility_bad_flips.csv", index=False)

    lines = [
        "# Utility Active Class Flip Analysis",
        "",
        "Ez az elemzes csak azokat a snapshotokat nezi, ahol a with-utility es no-utility modell 0.5 threshold mellett mas class-t prediktalt.",
        "A csoportok azt mutatjak, hogy milyen utility jel volt jelen az adott atbillenesnel.",
        "",
        "| csoport | sorok | utility jo | no-utility jo | kulonbseg | utility win rate | mean abs delta | mean delta | CT win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        if int(row["rows"]) == 0:
            continue
        lines.append(
            f"| `{row['group']}` | {int(row['rows'])} | "
            f"{int(row['with_utility_correct'])} | {int(row['no_utility_correct'])} | "
            f"{int(row['with_minus_no'])} | {row['with_win_rate']:.2%} | "
            f"{row['mean_abs_delta']:.6f} | {row['mean_delta']:.6f} | {row['ct_win_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Csoportok jelentese",
            "",
            "- `all_flips`: minden atbillenes.",
            "- `any_utility_nonzero`: barmilyen utility feature nem nulla, inventory is beleszamit.",
            "- `active_or_recent_utility`: aktiv smoke/inferno, last_5s utility, utility damage vagy flash duration latszik.",
            "- `strong_utility_action`: erosebb utility-helyzet, peldaul tobb aktiv smoke/inferno, tobb recent utility event, legalabb 10 damage, vagy flash effect.",
            "- `utility_damage`: volt utility sebzes az utolso 5 masodpercben.",
            "- `active_smoke_or_inferno`: van aktiv smoke vagy inferno.",
            "- `recent_utility_last_5s`: volt smoke/flash/he/molly az utolso 5 masodpercben.",
            "- `flash_effect`: flash duration ertek alapjan valaki vakitva volt.",
            "",
            "## Kimenetek",
            "",
            "- `utility_active_flip_rows.csv`: minden atbillenes utility aktivitasi oszlopokkal.",
            "- `top_utility_good_flips.csv`: legnagyobb jo iranyu atbillenesek.",
            "- `top_utility_bad_flips.csv`: legnagyobb rossz iranyu atbillenesek.",
        ]
    )
    (output_dir / "utility_active_flip_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    df["abs_delta"] = pd.to_numeric(df["abs_delta"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")

    groups = find_utility_columns(df.columns)
    enriched = add_activity_columns(df, groups)

    summary_rows = [
        summarize_mask(enriched, pd.Series(True, index=enriched.index), "all_flips"),
        summarize_mask(enriched, enriched["has_any_utility"], "any_utility_nonzero"),
        summarize_mask(enriched, enriched["has_active_or_recent_utility"], "active_or_recent_utility"),
        summarize_mask(enriched, enriched["strong_utility_action"], "strong_utility_action"),
        summarize_mask(enriched, enriched["has_utility_damage"], "utility_damage"),
        summarize_mask(enriched, enriched["has_active_smoke_or_inferno"], "active_smoke_or_inferno"),
        summarize_mask(enriched, enriched["has_recent_utility"], "recent_utility_last_5s"),
        summarize_mask(enriched, enriched["has_flash_effect"], "flash_effect"),
    ]
    summary = pd.DataFrame(summary_rows)
    write_summary(Path(args.output_dir), enriched, summary, args.top_k)
    print(f"Mentve: {args.output_dir}")


if __name__ == "__main__":
    main()
