import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from train_xgboost import is_utility_column


PREFERRED_CONTEXT_COLUMNS = (
    "round_num",
    "tick",
    "seconds_in_round",
    "ct_score",
    "t_score",
    "ct_alive",
    "t_alive",
    "alive_diff",
    "ct_hp",
    "t_hp",
    "hp_diff",
    "ct_equipment_value",
    "t_equipment_value",
    "equipment_value_diff",
    "bomb_planted",
    "bomb_site",
)

UTILITY_HINTS = (
    "last_5s",
    "active",
    "damage",
    "inv",
    "duration",
    "smoke",
    "flash",
    "molly",
    "inferno",
    "he",
)

UTILITY_REPORT_EXCLUDE = (
    "alpha_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5-10 konkret delta win probability pelda keszitese a utility/no-utility modellekhez."
    )
    parser.add_argument(
        "--delta-dir",
        type=str,
        required=True,
        help="A delta_win_probability.py kimeneti mappaja.",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Repo/project gyoker. Alapbol az aktualis mappa.",
    )
    parser.add_argument("--good-cases", type=int, default=6)
    parser.add_argument("--bad-cases", type=int, default=4)
    parser.add_argument(
        "--min-abs-delta",
        type=float,
        default=0.03,
        help="Minimum probability kulonbseg a valogatashoz.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="delta_case_studies",
    )
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def choose_cases(top_rows: pd.DataFrame, good_n: int, bad_n: int, min_abs_delta: float) -> pd.DataFrame:
    rows = top_rows[top_rows["abs_delta"] >= min_abs_delta].copy()
    rows["good_direction"] = as_bool(rows["good_direction"])
    rows["case_key"] = rows["csv_path"].astype(str) + " | round " + rows["round_num"].astype(str)

    good = (
        rows[rows["good_direction"]]
        .sort_values("abs_delta", ascending=False)
        .drop_duplicates("case_key")
        .head(good_n)
        .copy()
    )
    good["case_type"] = "utility jobb iranyba vitte"

    bad = (
        rows[~rows["good_direction"]]
        .sort_values("abs_delta", ascending=False)
        .drop_duplicates("case_key")
        .head(bad_n)
        .copy()
    )
    bad["case_type"] = "utility rosszabb iranyba vitte"

    return pd.concat([good, bad], ignore_index=True)


def csv_abs_path(project_dir: Path, rel_csv: str) -> Path:
    path = project_dir / "onlab" / rel_csv
    if path.exists():
        return path
    path = project_dir / rel_csv
    if path.exists():
        return path
    return project_dir / "onlab" / rel_csv.replace("\\", "/")


def parse_value(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            return float(text)
        except ValueError:
            return None
    if isinstance(parsed, (int, float)):
        return float(parsed)
    return None


def nonzero_utility_values(row: pd.Series, columns: Iterable[str], limit: int = 10) -> str:
    values: List[tuple[str, float]] = []
    for col in columns:
        value = parse_value(row.get(col))
        if value is None or abs(value) <= 1e-12:
            continue
        values.append((col, value))
    values.sort(key=lambda item: abs(item[1]), reverse=True)
    return "; ".join(f"{col}={value:g}" for col, value in values[:limit])


def utility_columns_for_report(columns: Iterable[str]) -> List[str]:
    selected = []
    for col in columns:
        lowered = col.lower()
        if not is_utility_column(col):
            continue
        if any(token in lowered for token in UTILITY_REPORT_EXCLUDE):
            continue
        if any(hint in lowered for hint in UTILITY_HINTS):
            selected.append(col)
    return selected


def load_original_rows(project_dir: Path, cases: pd.DataFrame) -> pd.DataFrame:
    enriched: List[Dict[str, object]] = []
    for rel_csv, group in cases.groupby("csv_path", sort=False):
        path = csv_abs_path(project_dir, rel_csv)
        original = pd.read_csv(path)
        utility_cols = utility_columns_for_report(original.columns)
        context_cols = [col for col in PREFERRED_CONTEXT_COLUMNS if col in original.columns]

        for _, case in group.iterrows():
            row_index = int(case["csv_row_index"])
            if row_index < 0 or row_index >= len(original):
                source = pd.Series(dtype=object)
            else:
                source = original.iloc[row_index]

            out: Dict[str, object] = case.to_dict()
            out["match_file"] = Path(str(rel_csv)).name
            out["top_utility_values"] = nonzero_utility_values(source, utility_cols)
            for col in context_cols:
                out[f"src_{col}"] = source.get(col)
            enriched.append(out)
    return pd.DataFrame(enriched)


def fmt_prob(value: float) -> str:
    return f"{float(value):.3f}"


def write_markdown(path: Path, cases: pd.DataFrame) -> None:
    lines = [
        "# Delta Case Studies",
        "",
        "Ezek a peldak a with-utility es no-utility modell CT win probability kulonbseget mutatjak.",
        "A `delta` = `p_with_utility - p_no_utility`; CT win eseten a pozitiv delta jo irany, T win eseten a negativ delta jo irany.",
        "",
    ]

    for _, row in cases.iterrows():
        direction = "jo irany" if bool(row["good_direction"]) else "rossz irany"
        title = (
            f"## {row['case_type']} - {row['match_file']} "
            f"round {row['round_num']}, {fmt_prob(row['seconds_in_round'])}s"
        )
        lines.extend(
            [
                title,
                "",
                f"- ct_win: `{int(row['ct_win'])}`",
                f"- p_with_utility: `{fmt_prob(row['p_with_utility'])}`",
                f"- p_no_utility: `{fmt_prob(row['p_no_utility'])}`",
                f"- delta: `{float(row['delta']):+.3f}` ({direction})",
                f"- logloss improvement: `{float(row['logloss_improvement']):+.3f}`",
                f"- csv: `{row['csv_path']}`",
            ]
        )
        top_values = str(row.get("top_utility_values", "") or "")
        if top_values:
            lines.append(f"- fontosabb utility ertekek: `{top_values}`")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    delta_dir = Path(args.delta_dir)
    project_dir = Path(args.project_dir)
    top_rows = pd.read_csv(delta_dir / "top_delta_rows.csv")

    cases = choose_cases(
        top_rows=top_rows,
        good_n=args.good_cases,
        bad_n=args.bad_cases,
        min_abs_delta=args.min_abs_delta,
    )
    enriched = load_original_rows(project_dir=project_dir, cases=cases)

    csv_path = delta_dir / f"{args.output_prefix}.csv"
    md_path = delta_dir / f"{args.output_prefix}.md"
    enriched.to_csv(csv_path, index=False)
    write_markdown(md_path, enriched)

    print(f"Case study CSV: {csv_path}")
    print(f"Case study MD: {md_path}")


if __name__ == "__main__":
    main()
