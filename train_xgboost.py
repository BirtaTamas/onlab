import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.utils import CSGOUtils


@dataclass
class CsvMeta:
    path: Path
    tournament: str
    series: str
    match_id: str
    map_name: str
    row_count: int


HIGH_CARDINALITY_ID_SUFFIXES = ("_name", "_steamid")
PLACE_CATEGORICAL_SUFFIXES = ("_place",)
WEAPON_CATEGORICAL_SUFFIXES = ("_primary_weapon", "_secondary_weapon")
USEFUL_CATEGORICAL_SUFFIXES = PLACE_CATEGORICAL_SUFFIXES + WEAPON_CATEGORICAL_SUFFIXES
DEFAULT_DROP_COLUMNS = {"ct_win"}
PARTIAL_CSV_SUFFIXES = ("-p1", "-p2")
UTILITY_TOKENS = {
    "utility",
    "smoke",
    "smokes",
    "flash",
    "flashes",
    "he",
    "molly",
    "mollies",
    "inferno",
    "infernos",
}
STRING_COLUMN_SUFFIXES = (
    "_name",
    "_steamid",
    "_place",
    "_primary_weapon",
    "_secondary_weapon",
)

PLAYER_SLOT_STRONG_RE = re.compile(r"^(T|CT)[1-5]__(alive|hp|armor|has_helmet)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match-level split with map-balance and XGBoost training."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Gyoker mappa, ahol a tournament/series/*.csv struktura talalhato.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/xgboost_run",
        help="Kimeneti mappa a split metadatahoz, modellhez es metrikakhoz.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train arany match szinten.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Validacios arany match szinten.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Teszt arany match szinten.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed a determinisztikus sorrendhez.",
    )
    parser.add_argument(
        "--use-library-defaults",
        action="store_true",
        help="Ha meg van adva, a nem explicit megadott XGBoost hyperparameterekhez a library defaultjai lesznek hasznalva.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="XGBoost max_depth.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="XGBoost n_estimators.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="XGBoost learning_rate.",
    )
    parser.add_argument(
        "--min-child-weight",
        type=float,
        default=5.0,
        help="XGBoost min_child_weight.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="XGBoost subsample.",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="XGBoost colsample_bytree.",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=2.0,
        help="XGBoost L2 regularizacio.",
    )
    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0.0,
        help="XGBoost L1 regularizacio.",
    )
    parser.add_argument(
        "--max-cat-to-onehot",
        type=int,
        default=16,
        help="Kis kategorias oszlopoknal egy-hot hatar XGBoost categorical modhoz.",
    )
    parser.add_argument(
        "--include-partial-csvs",
        action="store_true",
        help="Ha meg van adva, a -p1/-p2 resz-CSV-k is bekerulnek.",
    )
    parser.add_argument(
        "--sample-csv-ratio",
        type=float,
        default=1.0,
        help="Spliten belul map-aranyosan megtartott teljes CSV-k aranya (0-1].",
    )
    parser.add_argument(
        "--row-stride",
        type=int,
        default=1,
        help="Minden n-edik sort tartjuk meg a CSV-ken belul.",
    )
    parser.add_argument(
        "--include-categorical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="A hasznos kategorias oszlopok bekerulnek. Kikapcsolashoz: --no-include-categorical.",
    )
    parser.add_argument(
        "--include-place-categorical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="A *_place kategorias oszlopok bekerulnek. Kikapcsolashoz: --no-include-place-categorical.",
    )
    parser.add_argument(
        "--include-weapon-categorical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="A *_primary_weapon es *_secondary_weapon kategorias oszlopok bekerulnek. Kikapcsolashoz: --no-include-weapon-categorical.",
    )
    parser.add_argument(
        "--include-tick",
        action="store_true",
        help="Ha meg van adva, a tick oszlopot nem dobjuk el.",
    )
    parser.add_argument(
        "--drop-utility-features",
        action="store_true",
        help="Ha meg van adva, a utilityhez kotheto feature-ok kimaradnak.",
    )
    parser.add_argument(
        "--drop-strong-non-utility-features",
        action="store_true",
        help="Ha meg van adva, a nagyon eros nem-utility feature-ok (alive/hp/armor/economy/equip) kimaradnak.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Csak splitet es sample manifestet keszit, tanitas nelkul.",
    )
    return parser.parse_args()


def validate_split_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"A split aranyok osszege 1.0 kell legyen, most: {total:.6f}"
        )


def validate_sampling_args(sample_csv_ratio: float, row_stride: int) -> None:
    if not (0.0 < sample_csv_ratio <= 1.0):
        raise ValueError(
            f"A sample-csv-ratio erteke 0 es 1 kozott kell legyen, most: {sample_csv_ratio}"
        )
    if row_stride < 1:
        raise ValueError(f"A row-stride legalabb 1 kell legyen, most: {row_stride}")


def is_partial_csv(csv_path: Path) -> bool:
    return csv_path.stem.endswith(PARTIAL_CSV_SUFFIXES)


def discover_csv_files(data_root: Path, include_partial_csvs: bool) -> List[Path]:
    csv_files = sorted(data_root.rglob("*.csv"))
    if not include_partial_csvs:
        csv_files = [path for path in csv_files if not is_partial_csv(path)]
    if not csv_files:
        raise FileNotFoundError(f"Nem talaltam CSV fajlokat itt: {data_root}")
    return csv_files


def infer_match_id(csv_path: Path, data_root: Path) -> str:
    rel_path = csv_path.relative_to(data_root)
    return str(rel_path.with_suffix(""))


def infer_map_name(csv_path: Path) -> str:
    return CSGOUtils.infer_map_name(csv_path.stem, str(csv_path.parent), str(csv_path))


def collect_csv_metadata(data_root: Path, include_partial_csvs: bool) -> List[CsvMeta]:
    items: List[CsvMeta] = []

    for csv_path in discover_csv_files(data_root, include_partial_csvs=include_partial_csvs):
        rel_parts = csv_path.relative_to(data_root).parts
        tournament = rel_parts[0] if len(rel_parts) >= 1 else "unknown_tournament"
        series = rel_parts[1] if len(rel_parts) >= 2 else "unknown_series"
        row_count = (
            pl.scan_csv(str(csv_path))
            .select(pl.len().alias("row_count"))
            .collect()
            .item()
        )
        items.append(
            CsvMeta(
                path=csv_path,
                tournament=tournament,
                series=series,
                match_id=infer_match_id(csv_path, data_root),
                map_name=infer_map_name(csv_path),
                row_count=int(row_count),
            )
        )

    return items


def build_match_table(metadata: List[CsvMeta], seed: int) -> List[Dict]:
    grouped: Dict[str, Dict] = {}

    for item in metadata:
        group = grouped.setdefault(
            item.match_id,
            {
                "match_id": item.match_id,
                "paths": [],
                "map_counts": {},
                "row_count": 0,
            },
        )
        group["paths"].append(item.path)
        group["map_counts"][item.map_name] = group["map_counts"].get(item.map_name, 0) + item.row_count
        group["row_count"] += item.row_count

    rng = np.random.default_rng(seed)
    matches = list(grouped.values())
    rng.shuffle(matches)
    matches.sort(key=lambda row: row["row_count"], reverse=True)
    return matches


def dominant_map(match_row: Dict) -> str:
    map_counts = match_row["map_counts"]
    return max(map_counts, key=map_counts.get) if map_counts else "unknown"


def assign_matches_to_splits(
    matches: List[Dict],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Dict[str, List[Dict]]:
    split_names = ["train", "valid", "test"]
    target_ratios = {"train": train_ratio, "valid": valid_ratio, "test": test_ratio}
    total_rows = sum(match["row_count"] for match in matches)

    all_maps = sorted(
        {
            map_name
            for match in matches
            for map_name in match["map_counts"].keys()
        }
    )
    global_map_rows = {
        map_name: sum(match["map_counts"].get(map_name, 0) for match in matches)
        for map_name in all_maps
    }

    split_rows = {name: 0 for name in split_names}
    split_map_rows = {name: {map_name: 0 for map_name in all_maps} for name in split_names}
    assigned = {name: [] for name in split_names}

    for match in matches:
        best_split = None
        best_score = None

        for split_name in split_names:
            candidate_rows = split_rows.copy()
            candidate_rows[split_name] += match["row_count"]

            candidate_map_rows = {
                name: split_map_rows[name].copy() for name in split_names
            }
            for map_name, value in match["map_counts"].items():
                candidate_map_rows[split_name][map_name] += value

            size_penalty = 0.0
            for name in split_names:
                target_rows = total_rows * target_ratios[name]
                if total_rows > 0:
                    size_penalty += abs(candidate_rows[name] - target_rows) / total_rows

            map_penalty = 0.0
            for name in split_names:
                for map_name in all_maps:
                    global_rows_for_map = global_map_rows[map_name]
                    if global_rows_for_map <= 0:
                        continue
                    actual_ratio = candidate_map_rows[name][map_name] / global_rows_for_map
                    map_penalty += abs(actual_ratio - target_ratios[name])

            # Elsodleges cel: map-aranyok kozel tartsak a celhoz.
            score = map_penalty * 5.0 + size_penalty

            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        split_rows[best_split] += match["row_count"]
        for map_name, value in match["map_counts"].items():
            split_map_rows[best_split][map_name] += value
        assigned[best_split].append(match)

    return assigned


def build_split_manifest(split_matches: Dict[str, List[Dict]]) -> List[Dict]:
    rows: List[Dict] = []

    for split_name, matches in split_matches.items():
        for match in matches:
            primary_map = dominant_map(match)
            for csv_path in match["paths"]:
                rows.append(
                    {
                        "split": split_name,
                        "match_id": match["match_id"],
                        "primary_map_name": primary_map,
                        "csv_path": str(csv_path),
                        "match_row_count": match["row_count"],
                    }
                )

    return rows


def sample_manifest_rows(
    manifest_rows: List[Dict],
    sample_csv_ratio: float,
    seed: int,
) -> List[Dict]:
    if sample_csv_ratio >= 1.0:
        return manifest_rows

    rng = np.random.default_rng(seed)
    grouped: Dict[tuple, List[Dict]] = {}
    sampled_rows: List[Dict] = []

    for row in manifest_rows:
        key = (row["split"], row["primary_map_name"])
        grouped.setdefault(key, []).append(row)

    for key in sorted(grouped):
        rows = grouped[key]
        sample_size = max(1, int(round(len(rows) * sample_csv_ratio)))
        sample_size = min(sample_size, len(rows))
        selected_idx = sorted(rng.choice(len(rows), size=sample_size, replace=False).tolist())
        for idx in selected_idx:
            sampled_rows.append(rows[idx])

    return sampled_rows


def infer_schema_overrides(csv_path: Path) -> Dict[str, pl.DataType]:
    header = pl.read_csv(str(csv_path), n_rows=0)
    schema_overrides: Dict[str, pl.DataType] = {}

    for col_name in header.columns:
        if col_name.endswith(STRING_COLUMN_SUFFIXES):
            schema_overrides[col_name] = pl.Utf8

    return schema_overrides


def read_split_dataframe(csv_paths: List[Path], row_stride: int) -> pl.DataFrame:
    frames = [
        pl.read_csv(
            str(path),
            schema_overrides=infer_schema_overrides(path),
            infer_schema_length=10000,
        )
        for path in csv_paths
    ]
    if not frames:
        raise ValueError("Ures splithez nem lehet adatot betolteni.")
    if row_stride > 1:
        frames = [
            frame.with_row_index("__row_idx")
            .filter((pl.col("__row_idx") % row_stride) == 0)
            .drop("__row_idx")
            for frame in frames
        ]
    return pl.concat(frames, how="diagonal_relaxed")


def is_utility_column(col_name: str) -> bool:
    lowered = col_name.lower()
    tokens = set(lowered.split("_"))
    return any(token in UTILITY_TOKENS for token in tokens)


def is_strong_non_utility_column(col_name: str) -> bool:
    if is_utility_column(col_name):
        return False

    if col_name in {"T_alive", "CT_alive", "alive_diff"}:
        return True
    if col_name in {"T_hp_sum", "CT_hp_sum", "hp_diff"}:
        return True
    if col_name in {"T_armor_sum", "CT_armor_sum", "armor_diff"}:
        return True
    if col_name in {"T_helmet_count", "CT_helmet_count"}:
        return True
    if PLAYER_SLOT_STRONG_RE.match(col_name):
        return True

    economy_tokens = (
        "money_sum",
        "money_diff",
        "start_balance_sum",
        "cash_spent_round_sum",
        "equip_value_sum",
        "round_start_equip_sum",
        "equip_diff",
    )
    return any(token in col_name for token in economy_tokens)


def choose_feature_columns(
    df: pl.DataFrame,
    label_col: str = "ct_win",
    include_categorical: bool = False,
    include_place_categorical: bool = True,
    include_weapon_categorical: bool = True,
    include_tick: bool = False,
    drop_utility_features: bool = False,
    drop_strong_non_utility_features: bool = False,
) -> Dict[str, List[str]]:
    if label_col not in df.columns:
        raise ValueError(f"Nem talalhato a label oszlop: {label_col}")

    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    dropped_columns: List[str] = []

    for col_name, dtype in zip(df.columns, df.dtypes):
        if col_name in DEFAULT_DROP_COLUMNS or col_name == label_col:
            continue

        if not include_tick and col_name == "tick":
            dropped_columns.append(col_name)
            continue

        if drop_utility_features and is_utility_column(col_name):
            dropped_columns.append(col_name)
            continue

        if drop_strong_non_utility_features and is_strong_non_utility_column(col_name):
            dropped_columns.append(col_name)
            continue

        if col_name.endswith(HIGH_CARDINALITY_ID_SUFFIXES):
            dropped_columns.append(col_name)
            continue

        if dtype.is_numeric() or dtype == pl.Boolean:
            numeric_columns.append(col_name)
            continue

        if include_categorical:
            include_this_categorical = False
            if include_place_categorical and col_name.endswith(PLACE_CATEGORICAL_SUFFIXES):
                include_this_categorical = True
            if include_weapon_categorical and col_name.endswith(WEAPON_CATEGORICAL_SUFFIXES):
                include_this_categorical = True
            if include_this_categorical:
                categorical_columns.append(col_name)
                continue

        dropped_columns.append(col_name)

    if not numeric_columns and not categorical_columns:
        raise ValueError("Nem talaltam hasznalhato feature oszlopokat a tanitashoz.")

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "dropped": dropped_columns,
    }


def materialize_features(
    df: pl.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
    label_col: str = "ct_win",
):
    pandas_df = df.select(numeric_columns + categorical_columns + [label_col]).to_pandas()

    for col_name in numeric_columns:
        if pd.api.types.is_bool_dtype(pandas_df[col_name]):
            pandas_df[col_name] = pandas_df[col_name].astype(np.int8)
        pandas_df[col_name] = pandas_df[col_name].fillna(0)

    for col_name in categorical_columns:
        pandas_df[col_name] = pandas_df[col_name].fillna("UNKNOWN").astype("category")

    x = pandas_df[numeric_columns + categorical_columns]
    y = pandas_df[label_col].fillna(0).astype(np.int64).to_numpy()
    return x, y


def align_categorical_levels(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_columns: List[str],
):
    for col_name in categorical_columns:
        combined = pd.concat(
            [
                train_df[col_name].astype("string"),
                valid_df[col_name].astype("string"),
                test_df[col_name].astype("string"),
            ],
            axis=0,
        ).fillna("UNKNOWN")
        categories = pd.Index(sorted(combined.unique()))
        train_df[col_name] = pd.Categorical(train_df[col_name], categories=categories)
        valid_df[col_name] = pd.Categorical(valid_df[col_name], categories=categories)
        test_df[col_name] = pd.Categorical(test_df[col_name], categories=categories)


def drop_constant_numeric_columns(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    test_x: pd.DataFrame,
    numeric_columns: List[str],
):
    kept_numeric: List[str] = []
    for col_name in numeric_columns:
        if train_x[col_name].nunique(dropna=False) > 1:
            kept_numeric.append(col_name)

    kept_columns = kept_numeric + [
        col_name for col_name in train_x.columns if col_name not in numeric_columns
    ]
    return train_x[kept_columns], valid_x[kept_columns], test_x[kept_columns], kept_numeric


def prepare_datasets(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    test_df: pl.DataFrame,
    label_col: str = "ct_win",
    include_categorical: bool = False,
    include_place_categorical: bool = True,
    include_weapon_categorical: bool = True,
    include_tick: bool = False,
    drop_utility_features: bool = False,
    drop_strong_non_utility_features: bool = False,
):
    feature_plan = choose_feature_columns(
        train_df,
        label_col=label_col,
        include_categorical=include_categorical,
        include_place_categorical=include_place_categorical,
        include_weapon_categorical=include_weapon_categorical,
        include_tick=include_tick,
        drop_utility_features=drop_utility_features,
        drop_strong_non_utility_features=drop_strong_non_utility_features,
    )
    numeric_columns = feature_plan["numeric"]
    categorical_columns = feature_plan["categorical"]

    train_x, train_y = materialize_features(
        train_df, numeric_columns, categorical_columns, label_col=label_col
    )
    valid_x, valid_y = materialize_features(
        valid_df, numeric_columns, categorical_columns, label_col=label_col
    )
    test_x, test_y = materialize_features(
        test_df, numeric_columns, categorical_columns, label_col=label_col
    )

    if categorical_columns:
        align_categorical_levels(train_x, valid_x, test_x, categorical_columns)

    train_x, valid_x, test_x, numeric_columns = drop_constant_numeric_columns(
        train_x, valid_x, test_x, numeric_columns
    )

    used_columns = list(train_x.columns)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "valid_x": valid_x,
        "valid_y": valid_y,
        "test_x": test_x,
        "test_y": test_y,
        "feature_names": used_columns,
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "dropped_features": feature_plan["dropped"],
    }


def summarize_split(split_name: str, matches: List[Dict]) -> Dict:
    map_rows: Dict[str, int] = {}
    total_rows = 0

    for match in matches:
        total_rows += match["row_count"]
        for map_name, row_count in match["map_counts"].items():
            map_rows[map_name] = map_rows.get(map_name, 0) + row_count

    return {
        "split": split_name,
        "match_count": len(matches),
        "row_count": total_rows,
        "map_rows": map_rows,
    }


def summarize_manifest_rows(split_name: str, manifest_rows: List[Dict]) -> Dict:
    map_csv_counts: Dict[str, int] = {}
    for row in manifest_rows:
        if row["split"] != split_name:
            continue
        map_name = row["primary_map_name"]
        map_csv_counts[map_name] = map_csv_counts.get(map_name, 0) + 1

    return {
        "split": split_name,
        "csv_count": sum(map_csv_counts.values()),
        "map_csv_counts": map_csv_counts,
    }


def train_model(
    x_train,
    y_train,
    x_valid,
    y_valid,
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    min_child_weight: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    reg_alpha: float,
    max_cat_to_onehot: int,
    seed: int,
    use_library_defaults: bool = False,
):
    from xgboost import XGBClassifier

    pos_count = max(int((y_train == 1).sum()), 1)
    neg_count = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = neg_count / pos_count
    has_categorical_features = any(str(dtype) == "category" for dtype in x_train.dtypes)

    model_kwargs = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )
    if use_library_defaults:
        model_kwargs.update(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
        if has_categorical_features:
            model_kwargs.update(
                tree_method="hist",
                enable_categorical=True,
            )
    else:
        model_kwargs.update(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            max_cat_to_onehot=max_cat_to_onehot,
            tree_method="hist",
            enable_categorical=True,
            n_jobs=-1,
        )

    model = XGBClassifier(**model_kwargs)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=False,
    )
    return model


def build_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, object]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Az 1.0 valoszinusegu mintak az utolso binbe keruljenek.
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=False)

    curve_rows = []
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        count = int(mask.sum())
        if count == 0:
            true_rate = None
            mean_pred = None
        else:
            true_rate = float(np.mean(y_true[mask]))
            mean_pred = float(np.mean(y_prob[mask]))

        curve_rows.append(
            {
                "bin_index": bin_idx,
                "bin_start": float(bin_edges[bin_idx]),
                "bin_end": float(bin_edges[bin_idx + 1]),
                "count": count,
                "mean_predicted_probability": mean_pred,
                "fraction_of_positives": true_rate,
            }
        )

    return {
        "n_bins": n_bins,
        "strategy": "uniform",
        "bins": curve_rows,
    }


def evaluate_binary_classifier(model, x_data, y_true) -> Dict[str, object]:
    y_prob = model.predict_proba(x_data)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(np.mean((y_prob - y_true) ** 2)),
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "threshold": 0.5,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": [
                [int(tn), int(fp)],
                [int(fn), int(tp)],
            ],
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "calibration_curve": build_calibration_curve(
            y_true=np.asarray(y_true),
            y_prob=y_prob,
        ),
    }

    unique_labels = np.unique(y_true)
    if len(unique_labels) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None

    return metrics


def save_evaluation_plots(
    output_dir: Path,
    split_name: str,
    split_metrics: Dict[str, object],
) -> Dict[str, str]:
    mpl_config_dir = output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib.pyplot as plt

    confusion = split_metrics["confusion_matrix"]
    matrix = np.asarray(confusion["matrix"])

    confusion_path = output_dir / f"{split_name}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{split_name.capitalize()} confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])

    max_value = matrix.max() if matrix.size else 0
    threshold = max_value / 2 if max_value else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            text_color = "white" if value > threshold else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=text_color)

    fig.tight_layout()
    fig.savefig(confusion_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    calibration = split_metrics["calibration_curve"]
    used_bins = [
        row
        for row in calibration["bins"]
        if row["count"] > 0
        and row["mean_predicted_probability"] is not None
        and row["fraction_of_positives"] is not None
    ]

    calibration_path = output_dir / f"{split_name}_calibration_curve.png"
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Perfect calibration")
    if used_bins:
        ax.plot(
            [row["mean_predicted_probability"] for row in used_bins],
            [row["fraction_of_positives"] for row in used_bins],
            marker="o",
            color="#1f77b4",
            label=split_name.capitalize(),
        )
    ax.set_title(f"{split_name.capitalize()} calibration curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(calibration_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "confusion_matrix_plot": str(confusion_path),
        "calibration_curve_plot": str(calibration_path),
    }


def main() -> None:
    args = parse_args()
    validate_split_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
    validate_sampling_args(args.sample_csv_ratio, args.row_stride)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = collect_csv_metadata(
        data_root,
        include_partial_csvs=args.include_partial_csvs,
    )
    matches = build_match_table(metadata, seed=args.random_seed)
    split_matches = assign_matches_to_splits(
        matches,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    manifest_rows = build_split_manifest(split_matches)
    manifest_df = pl.DataFrame(manifest_rows)
    manifest_path = output_dir / "split_manifest.csv"
    manifest_df.write_csv(str(manifest_path))
    sampled_manifest_rows = sample_manifest_rows(
        manifest_rows,
        sample_csv_ratio=args.sample_csv_ratio,
        seed=args.random_seed,
    )
    sampled_manifest_df = pl.DataFrame(sampled_manifest_rows)
    sampled_manifest_path = output_dir / "sampled_split_manifest.csv"
    sampled_manifest_df.write_csv(str(sampled_manifest_path))

    split_summaries = [
        summarize_split(split_name, split_matches[split_name])
        for split_name in ["train", "valid", "test"]
    ]
    sampled_split_summaries = [
        summarize_manifest_rows(split_name, sampled_manifest_rows)
        for split_name in ["train", "valid", "test"]
    ]

    train_paths = [Path(row["csv_path"]) for row in sampled_manifest_rows if row["split"] == "train"]
    valid_paths = [Path(row["csv_path"]) for row in sampled_manifest_rows if row["split"] == "valid"]
    test_paths = [Path(row["csv_path"]) for row in sampled_manifest_rows if row["split"] == "test"]

    if not train_paths or not valid_paths or not test_paths:
        raise ValueError(
            "Legalabb egy split ures lett. Erdemes tobb meccsel dolgozni vagy mas split aranyt adni."
        )

    if args.prepare_only:
        prep_summary = {
            "sampling": {
                "include_partial_csvs": args.include_partial_csvs,
                "sample_csv_ratio": args.sample_csv_ratio,
                "row_stride": args.row_stride,
            },
            "splits": split_summaries,
            "sampled_splits": sampled_split_summaries,
        }
        prep_summary_path = output_dir / "prepare_summary.json"
        prep_summary_path.write_text(json.dumps(prep_summary, indent=2), encoding="utf-8")
        print(f"Split manifest mentve: {manifest_path}")
        print(f"Mintavetelezett manifest mentve: {sampled_manifest_path}")
        print(f"Osszegzes mentve: {prep_summary_path}")
        return

    train_df = read_split_dataframe(train_paths, row_stride=args.row_stride)
    valid_df = read_split_dataframe(valid_paths, row_stride=args.row_stride)
    test_df = read_split_dataframe(test_paths, row_stride=args.row_stride)

    prepared = prepare_datasets(
        train_df,
        valid_df,
        test_df,
        label_col="ct_win",
        include_categorical=args.include_categorical,
        include_place_categorical=args.include_place_categorical,
        include_weapon_categorical=args.include_weapon_categorical,
        include_tick=args.include_tick,
        drop_utility_features=args.drop_utility_features,
        drop_strong_non_utility_features=args.drop_strong_non_utility_features,
    )
    x_train = prepared["train_x"]
    y_train = prepared["train_y"]
    x_valid = prepared["valid_x"]
    y_valid = prepared["valid_y"]
    x_test = prepared["test_x"]
    y_test = prepared["test_y"]
    feature_names = prepared["feature_names"]

    model = train_model(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        max_cat_to_onehot=args.max_cat_to_onehot,
        seed=args.random_seed,
        use_library_defaults=args.use_library_defaults,
    )

    split_metrics = {
        "train": evaluate_binary_classifier(model, x_train, y_train),
        "valid": evaluate_binary_classifier(model, x_valid, y_valid),
        "test": evaluate_binary_classifier(model, x_test, y_test),
    }

    plot_artifacts = {}
    for split_name, split_metric_values in split_metrics.items():
        plot_artifacts[split_name] = save_evaluation_plots(
            output_dir=output_dir,
            split_name=split_name,
            split_metrics=split_metric_values,
        )

    metrics = {
        "train": split_metrics["train"],
        "valid": split_metrics["valid"],
        "test": split_metrics["test"],
        "plot_artifacts": plot_artifacts,
        "feature_count": len(feature_names),
        "features": feature_names,
        "numeric_feature_count": len(prepared["numeric_features"]),
        "categorical_feature_count": len(prepared["categorical_features"]),
        "categorical_features": prepared["categorical_features"],
        "dropped_features": prepared["dropped_features"],
        "splits": split_summaries,
        "sampled_splits": sampled_split_summaries,
        "sampling": {
            "include_partial_csvs": args.include_partial_csvs,
            "sample_csv_ratio": args.sample_csv_ratio,
            "row_stride": args.row_stride,
        },
        "feature_flags": {
            "include_categorical": args.include_categorical,
            "include_place_categorical": args.include_place_categorical,
            "include_weapon_categorical": args.include_weapon_categorical,
            "include_tick": args.include_tick,
            "drop_utility_features": args.drop_utility_features,
            "drop_strong_non_utility_features": args.drop_strong_non_utility_features,
        },
        "xgboost_defaults_mode": "library" if args.use_library_defaults else "script",
    }

    model_path = output_dir / "xgboost_model.json"
    metrics_path = output_dir / "metrics.json"
    model.save_model(str(model_path))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Split manifest mentve: {manifest_path}")
    print(f"Modell mentve: {model_path}")
    print(f"Metrikak mentve: {metrics_path}")
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
