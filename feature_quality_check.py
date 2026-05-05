import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


UTILITY_TOKENS = (
    "utility",
    "smoke",
    "flash",
    "he",
    "molly",
    "inferno",
)
FORBIDDEN_PATTERNS = (
    "ct_win",
    "winner",
    "round_winner",
    "steamid",
    "_name",
    "__place",
    "__primary_weapon",
    "__secondary_weapon",
)
LEAK_SUSPICIOUS_TOKENS = (
    "winner",
    "round_winner",
    "round_end",
    "end_reason",
    "result",
    "future",
    "next",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature quality es leakage sanity check egy modellfutasra."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Modellfutas mappa, ahol metrics.json es sampled_split_manifest.csv van.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Opcionális adatgyoker. Ha nincs megadva, a run-dir alapjan probaljuk kitalalni.",
    )
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="train")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--rare-threshold", type=float, default=0.001)
    parser.add_argument("--unique-limit", type=int, default=10000)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_project_paths(run_dir: Path, data_root_arg: str | None) -> tuple[Path, Path]:
    if data_root_arg:
        data_root = Path(data_root_arg)
    else:
        # run_dir = onlab/artifacts/modellfutasok/<run>
        onlab_root = run_dir.parents[2]
        data_root = onlab_root / "processed_full"
    return data_root, run_dir / "sampled_split_manifest.csv"


def manifest_paths(manifest_path: Path, data_root: Path, split: str) -> List[Path]:
    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split != "all" and row["split"] != split:
                continue
            rows.append(data_root.parent / row["csv_path"])
    return rows


def classify_feature(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in UTILITY_TOKENS):
        return "utility"
    if any(token in lowered for token in ("alive", "hp", "armor", "helmet")):
        return "state"
    if any(token in lowered for token in ("money", "cash", "equip", "balance")):
        return "economy"
    if any(token in lowered for token in ("place", "macro", "mean_x", "mean_y", "spread", "dist", "_x", "_y")):
        return "position"
    if any(token in lowered for token in ("damage", "kill", "shots")):
        return "combat"
    if any(token in lowered for token in ("bomb", "defus")):
        return "objective"
    if "__" in name:
        return "player_slot_numeric"
    if any(token in lowered for token in ("round", "seconds")):
        return "time"
    return "other"


def is_forbidden_feature(name: str) -> bool:
    lowered = name.lower()
    return any(pattern in lowered for pattern in FORBIDDEN_PATTERNS)


def is_leak_suspicious(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in LEAK_SUSPICIOUS_TOKENS)


def new_stats(features: Iterable[str]) -> Dict[str, dict]:
    return {
        feature: {
            "feature": feature,
            "block": classify_feature(feature),
            "rows": 0,
            "non_null": 0,
            "zero": 0,
            "nonzero": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": math.inf,
            "max": -math.inf,
            "unique_values": set(),
            "unique_overflow": False,
            "forbidden": is_forbidden_feature(feature),
            "leak_suspicious": is_leak_suspicious(feature),
        }
        for feature in features
    }


def update_stats(stats: Dict[str, dict], chunk: pd.DataFrame, unique_limit: int) -> None:
    row_count = len(chunk)
    for feature, item in stats.items():
        if feature not in chunk.columns:
            values = pd.Series(np.zeros(row_count), index=chunk.index)
        else:
            values = pd.to_numeric(chunk[feature], errors="coerce").fillna(0)

        arr = values.to_numpy()
        item["rows"] += row_count
        item["non_null"] += int(values.notna().sum())
        zero_count = int((arr == 0).sum())
        item["zero"] += zero_count
        item["nonzero"] += int(row_count - zero_count)
        item["sum"] += float(arr.sum())
        item["sum_sq"] += float((arr * arr).sum())
        if row_count:
            item["min"] = min(item["min"], float(arr.min()))
            item["max"] = max(item["max"], float(arr.max()))

        if not item["unique_overflow"]:
            unique_values = pd.unique(values)
            for value in unique_values:
                item["unique_values"].add(float(value))
                if len(item["unique_values"]) > unique_limit:
                    item["unique_values"].clear()
                    item["unique_overflow"] = True
                    break


def read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readline().strip().split(",")


def scan_feature_stats(
    csv_paths: List[Path],
    features: List[str],
    chunk_size: int,
    unique_limit: int,
) -> Dict[str, dict]:
    stats = new_stats(features)
    needed = set(features)
    for index, csv_path in enumerate(csv_paths, start=1):
        if index % 25 == 0 or index == 1:
            print(f"CSV scan {index}/{len(csv_paths)}: {csv_path.name}")
        header = read_header(csv_path)
        usecols = [col for col in header if col in needed]
        if not usecols:
            continue
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
            update_stats(stats, chunk, unique_limit=unique_limit)
    return stats


def finalize_stats(stats: Dict[str, dict], rare_threshold: float) -> List[dict]:
    rows = []
    for feature, item in stats.items():
        n = item["rows"]
        mean = item["sum"] / n if n else 0.0
        variance = max(item["sum_sq"] / n - mean * mean, 0.0) if n else 0.0
        std = math.sqrt(variance)
        unique_count = None if item["unique_overflow"] else len(item["unique_values"])
        nonzero_ratio = item["nonzero"] / n if n else 0.0
        constant = (unique_count == 1) if unique_count is not None else False
        rare = (0.0 < nonzero_ratio < rare_threshold)
        rows.append(
            {
                "feature": feature,
                "block": item["block"],
                "rows": n,
                "non_null": item["non_null"],
                "zero": item["zero"],
                "nonzero": item["nonzero"],
                "nonzero_ratio": nonzero_ratio,
                "unique_count": unique_count,
                "unique_overflow": item["unique_overflow"],
                "min": 0.0 if item["min"] == math.inf else item["min"],
                "max": 0.0 if item["max"] == -math.inf else item["max"],
                "mean": mean,
                "std": std,
                "constant": constant,
                "rare_nonzero": rare,
                "forbidden": item["forbidden"],
                "leak_suspicious": item["leak_suspicious"],
            }
        )
    return rows


def write_report(run_dir: Path, rows: List[dict], metrics: dict, split: str) -> None:
    df = pd.DataFrame(rows).sort_values(["forbidden", "leak_suspicious", "constant", "rare_nonzero", "feature"], ascending=[False, False, False, False, True])
    csv_path = run_dir / f"feature_quality_{split}.csv"
    json_path = run_dir / f"feature_quality_{split}.json"
    md_path = run_dir / f"feature_quality_{split}.md"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    forbidden = df[df["forbidden"]]
    leak = df[df["leak_suspicious"]]
    constant = df[df["constant"]]
    rare = df[df["rare_nonzero"]]
    block_counts = df.groupby("block")["feature"].count().sort_values(ascending=False)
    utility_df = df[df["block"] == "utility"].sort_values("nonzero_ratio")

    lines = [
        "# Feature Quality Report",
        "",
        f"- run: `{run_dir.name}`",
        f"- split scanned: `{split}`",
        f"- feature count in metrics: `{metrics.get('feature_count')}`",
        f"- scanned feature count: `{len(df)}`",
        f"- forbidden hits: `{len(forbidden)}`",
        f"- leak-suspicious hits: `{len(leak)}`",
        f"- constant features: `{len(constant)}`",
        f"- rare nonzero features: `{len(rare)}`",
        "",
        "## Block Counts",
        "",
    ]
    for block, count in block_counts.items():
        lines.append(f"- `{block}`: `{count}`")

    lines.extend(["", "## Forbidden Hits", ""])
    if forbidden.empty:
        lines.append("- none")
    else:
        for _, row in forbidden.iterrows():
            lines.append(f"- `{row['feature']}`")

    lines.extend(["", "## Leak Suspicious Hits", ""])
    if leak.empty:
        lines.append("- none")
    else:
        for _, row in leak.iterrows():
            lines.append(f"- `{row['feature']}`")

    lines.extend(["", "## Constant Features", ""])
    if constant.empty:
        lines.append("- none")
    else:
        for _, row in constant.head(100).iterrows():
            lines.append(f"- `{row['feature']}`")
        if len(constant) > 100:
            lines.append(f"- ... plus `{len(constant) - 100}` more")

    lines.extend(["", "## Rare Nonzero Features", ""])
    if rare.empty:
        lines.append("- none")
    else:
        for _, row in rare.head(100).iterrows():
            lines.append(
                f"- `{row['feature']}` nonzero_ratio=`{row['nonzero_ratio']:.6f}` block=`{row['block']}`"
            )
        if len(rare) > 100:
            lines.append(f"- ... plus `{len(rare) - 100}` more")

    lines.extend(["", "## Rarest Utility Features", ""])
    if utility_df.empty:
        lines.append("- none")
    else:
        for _, row in utility_df.head(50).iterrows():
            lines.append(
                f"- `{row['feature']}` nonzero_ratio=`{row['nonzero_ratio']:.6f}` unique_count=`{row['unique_count']}`"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        alt_metrics = run_dir / "best_metrics.json"
        if alt_metrics.exists():
            metrics_path = alt_metrics
    metrics = load_json(metrics_path)
    features = metrics["features"]
    data_root, manifest_path = infer_project_paths(run_dir, args.data_root)
    paths = manifest_paths(manifest_path, data_root, split=args.split)
    if not paths:
        raise ValueError(f"Nincs CSV path ehhez a splithez: {args.split}")

    print(f"Run dir: {run_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Data root: {data_root}")
    print(f"Split: {args.split}, CSV count: {len(paths)}, feature count: {len(features)}")

    stats = scan_feature_stats(
        csv_paths=paths,
        features=features,
        chunk_size=args.chunk_size,
        unique_limit=args.unique_limit,
    )
    rows = finalize_stats(stats, rare_threshold=args.rare_threshold)
    write_report(run_dir, rows, metrics=metrics, split=args.split)


if __name__ == "__main__":
    main()
