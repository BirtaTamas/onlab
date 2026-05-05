import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from train_xgboost import (
    STRING_COLUMN_SUFFIXES,
    assign_matches_to_splits,
    build_calibration_curve,
    build_match_table,
    build_split_manifest,
    collect_csv_metadata,
    is_strong_non_utility_column,
    is_utility_column,
    sample_manifest_rows,
    save_evaluation_plots,
    summarize_manifest_rows,
    summarize_split,
    validate_sampling_args,
    validate_split_ratios,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


LABEL_COL = "ct_win"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAM-barat, streaming XGBoost tanitas CSV chunkokbol."
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/modellfutasok/xgboost_streaming_fixed_best_50pct",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--include-partial-csvs", action="store_true")
    parser.add_argument("--sample-csv-ratio", type=float, default=0.5)
    parser.add_argument("--row-stride", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--include-tick", action="store_true")
    parser.add_argument("--drop-utility-features", action="store_true")
    parser.add_argument(
        "--drop-flash-utility-features",
        action="store_true",
        help="Csak a flash jellegu utility feature-ok kidobasa, a tobbi utility bent marad.",
    )
    parser.add_argument("--drop-strong-non-utility-features", action="store_true")
    parser.add_argument(
        "--drop-constant-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trainen konstans feature-ok kidobasa. Alapbol bekapcsolva.",
    )

    # Defaultok: a 30%-os random search legjobb parameterpontja.
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--min-child-weight", type=float, default=10.0)
    parser.add_argument("--subsample", type=float, default=0.7802699952311001)
    parser.add_argument("--colsample-bytree", type=float, default=0.689587781698661)
    parser.add_argument("--reg-lambda", type=float, default=10.23671627286839)
    parser.add_argument("--reg-alpha", type=float, default=0.3292156351640363)
    parser.add_argument("--gamma", type=float, default=0.22455309696164494)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="XGBoost futtatas eszkoze. GPU-hoz: --device cuda.",
    )
    return parser.parse_args()


def read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readline().strip().split(",")


def is_streaming_feature_column(
    col_name: str,
    include_tick: bool,
    drop_utility_features: bool,
    drop_flash_utility_features: bool,
    drop_strong_non_utility_features: bool,
) -> bool:
    if col_name == LABEL_COL:
        return False
    if col_name == "tick" and not include_tick:
        return False
    if col_name.endswith(STRING_COLUMN_SUFFIXES):
        return False
    if drop_utility_features and is_utility_column(col_name):
        return False
    if drop_flash_utility_features and is_flash_utility_column(col_name):
        return False
    if drop_strong_non_utility_features and is_strong_non_utility_column(col_name):
        return False
    return True


def is_flash_utility_column(col_name: str) -> bool:
    lowered = col_name.lower()
    if not is_utility_column(col_name):
        return False
    return "flash" in lowered


def collect_feature_columns(
    csv_paths: Iterable[Path],
    include_tick: bool,
    drop_utility_features: bool,
    drop_flash_utility_features: bool,
    drop_strong_non_utility_features: bool,
) -> List[str]:
    features = set()
    for csv_path in csv_paths:
        for col_name in read_header(csv_path):
            if is_streaming_feature_column(
                col_name=col_name,
                include_tick=include_tick,
                drop_utility_features=drop_utility_features,
                drop_flash_utility_features=drop_flash_utility_features,
                drop_strong_non_utility_features=drop_strong_non_utility_features,
            ):
                features.add(col_name)
    return sorted(features)


def drop_constant_feature_columns(
    csv_paths: Iterable[Path],
    feature_columns: List[str],
    chunk_size: int,
    row_stride: int,
) -> Tuple[List[str], List[str]]:
    first_values = {}
    varying = {col: False for col in feature_columns}

    for x, _ in stream_csv_chunks(
        csv_paths=csv_paths,
        feature_columns=feature_columns,
        chunk_size=chunk_size,
        row_stride=row_stride,
    ):
        for col in feature_columns:
            if varying[col] or x.empty:
                continue
            values = x[col]
            if col not in first_values:
                first_values[col] = values.iloc[0]
            if (values != first_values[col]).any():
                varying[col] = True

    kept = [col for col in feature_columns if varying[col]]
    dropped = [col for col in feature_columns if not varying[col]]
    return kept, dropped


def stream_csv_chunks(
    csv_paths: Iterable[Path],
    feature_columns: List[str],
    chunk_size: int,
    row_stride: int,
) -> Iterator[Tuple[pd.DataFrame, np.ndarray]]:
    needed = set(feature_columns + [LABEL_COL])
    for csv_path in csv_paths:
        header = read_header(csv_path)
        usecols = [col for col in header if col in needed]
        if LABEL_COL not in usecols:
            raise ValueError(f"Hianyzik a label oszlop: {csv_path}")

        reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size)
        for chunk in reader:
            if row_stride > 1:
                chunk = chunk.iloc[::row_stride].copy()
            if chunk.empty:
                continue
            y = chunk[LABEL_COL].fillna(0).astype(np.int64).to_numpy()
            x = chunk.reindex(columns=feature_columns, fill_value=0)
            x = x.fillna(0)
            yield x, y


class CsvBatchIter(xgb.DataIter):
    def __init__(
        self,
        csv_paths: List[Path],
        feature_columns: List[str],
        chunk_size: int,
        row_stride: int,
    ):
        super().__init__()
        self.csv_paths = csv_paths
        self.feature_columns = feature_columns
        self.chunk_size = chunk_size
        self.row_stride = row_stride
        self._iterator = None

    def reset(self) -> None:
        self._iterator = stream_csv_chunks(
            csv_paths=self.csv_paths,
            feature_columns=self.feature_columns,
            chunk_size=self.chunk_size,
            row_stride=self.row_stride,
        )

    def next(self, input_data) -> int:
        if self._iterator is None:
            self.reset()
        try:
            x, y = next(self._iterator)
        except StopIteration:
            return 0
        input_data(data=x, label=y)
        return 1


def count_labels(
    csv_paths: Iterable[Path],
    chunk_size: int,
    row_stride: int,
) -> Tuple[int, int]:
    neg_count = 0
    pos_count = 0
    for csv_path in csv_paths:
        reader = pd.read_csv(csv_path, usecols=[LABEL_COL], chunksize=chunk_size)
        for chunk in reader:
            if row_stride > 1:
                chunk = chunk.iloc[::row_stride]
            labels = chunk[LABEL_COL].fillna(0).astype(np.int64)
            pos_count += int((labels == 1).sum())
            neg_count += int((labels == 0).sum())
    return neg_count, pos_count


def make_xgboost_params(args: argparse.Namespace, scale_pos_weight: float) -> Dict[str, object]:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": args.max_depth,
        "eta": args.learning_rate,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "lambda": args.reg_lambda,
        "alpha": args.reg_alpha,
        "gamma": args.gamma,
        "scale_pos_weight": scale_pos_weight,
        "seed": args.random_seed,
    }

    major_version = int(xgb.__version__.split(".", maxsplit=1)[0])
    if args.device == "cuda":
        if major_version >= 2:
            params.update({"tree_method": "hist", "device": "cuda"})
        else:
            params.update({"tree_method": "gpu_hist"})
    else:
        params.update({"tree_method": "hist"})
        if args.device == "cpu" and major_version >= 2:
            params.update({"device": "cpu"})
    return params


def predict_streaming(
    booster: xgb.Booster,
    csv_paths: List[Path],
    feature_columns: List[str],
    chunk_size: int,
    row_stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y_values = []
    prob_values = []
    best_iteration = getattr(booster, "best_iteration", None)
    iteration_range = None
    if best_iteration is not None and best_iteration >= 0:
        iteration_range = (0, int(best_iteration) + 1)

    for x, y in stream_csv_chunks(
        csv_paths=csv_paths,
        feature_columns=feature_columns,
        chunk_size=chunk_size,
        row_stride=row_stride,
    ):
        dmat = xgb.DMatrix(x)
        if iteration_range is None:
            probs = booster.predict(dmat)
        else:
            probs = booster.predict(dmat, iteration_range=iteration_range)
        y_values.append(y)
        prob_values.append(probs)

    return np.concatenate(y_values), np.concatenate(prob_values)


def evaluate_from_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
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
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "calibration_curve": build_calibration_curve(y_true=y_true, y_prob=y_prob),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


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
    sampled_manifest_rows = sample_manifest_rows(
        manifest_rows,
        sample_csv_ratio=args.sample_csv_ratio,
        seed=args.random_seed,
    )

    pd.DataFrame(manifest_rows).to_csv(output_dir / "split_manifest.csv", index=False)
    pd.DataFrame(sampled_manifest_rows).to_csv(
        output_dir / "sampled_split_manifest.csv",
        index=False,
    )

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
        raise ValueError("Legalabb egy split ures lett.")

    print("Feature oszlopok osszegyujtese...")
    feature_columns = collect_feature_columns(
        csv_paths=train_paths,
        include_tick=args.include_tick,
        drop_utility_features=args.drop_utility_features,
        drop_flash_utility_features=args.drop_flash_utility_features,
        drop_strong_non_utility_features=args.drop_strong_non_utility_features,
    )
    dropped_constant_features: List[str] = []
    if args.drop_constant_features:
        print("Konstans feature-ok szurese train chunkok alapjan...")
        feature_columns, dropped_constant_features = drop_constant_feature_columns(
            csv_paths=train_paths,
            feature_columns=feature_columns,
            chunk_size=args.chunk_size,
            row_stride=args.row_stride,
        )
    print(f"Feature count: {len(feature_columns)}")
    print(f"Dropped constant feature count: {len(dropped_constant_features)}")

    print("Label arany szamolasa...")
    neg_count, pos_count = count_labels(
        train_paths,
        chunk_size=args.chunk_size,
        row_stride=args.row_stride,
    )
    scale_pos_weight = max(neg_count, 1) / max(pos_count, 1)

    train_iter = CsvBatchIter(
        csv_paths=train_paths,
        feature_columns=feature_columns,
        chunk_size=args.chunk_size,
        row_stride=args.row_stride,
    )
    valid_iter = CsvBatchIter(
        csv_paths=valid_paths,
        feature_columns=feature_columns,
        chunk_size=args.chunk_size,
        row_stride=args.row_stride,
    )

    print("QuantileDMatrix epites streaming modban...")
    dtrain = xgb.QuantileDMatrix(train_iter)
    dvalid = xgb.QuantileDMatrix(valid_iter, ref=dtrain)

    params = make_xgboost_params(args, scale_pos_weight=scale_pos_weight)
    print("Modell tanitasa...")
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=25,
    )

    model_path = output_dir / "xgboost_model.json"
    booster.save_model(str(model_path))

    print("Metrikak szamolasa streaming predikcioval...")
    split_metrics = {}
    for split_name, paths in [
        ("train", train_paths),
        ("valid", valid_paths),
        ("test", test_paths),
    ]:
        y_true, y_prob = predict_streaming(
            booster=booster,
            csv_paths=paths,
            feature_columns=feature_columns,
            chunk_size=args.chunk_size,
            row_stride=args.row_stride,
        )
        split_metrics[split_name] = evaluate_from_probabilities(y_true, y_prob)

    plot_artifacts = {}
    for split_name, split_metric_values in split_metrics.items():
        plot_artifacts[split_name] = save_evaluation_plots(
            output_dir=output_dir,
            split_name=split_name,
            split_metrics=split_metric_values,
        )

    metrics = {
        "params": params,
        "best_iteration": int(getattr(booster, "best_iteration", -1)),
        "best_score": float(getattr(booster, "best_score", np.nan)),
        "train": split_metrics["train"],
        "valid": split_metrics["valid"],
        "test": split_metrics["test"],
        "plot_artifacts": plot_artifacts,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "dropped_constant_feature_count": len(dropped_constant_features),
        "dropped_constant_features": dropped_constant_features,
        "splits": split_summaries,
        "sampled_splits": sampled_split_summaries,
        "sampling": {
            "include_partial_csvs": args.include_partial_csvs,
            "sample_csv_ratio": args.sample_csv_ratio,
            "row_stride": args.row_stride,
            "chunk_size": args.chunk_size,
        },
        "feature_flags": {
            "include_tick": args.include_tick,
            "drop_utility_features": args.drop_utility_features,
            "drop_flash_utility_features": args.drop_flash_utility_features,
            "drop_strong_non_utility_features": args.drop_strong_non_utility_features,
            "drop_constant_features": args.drop_constant_features,
        },
        "label_counts": {
            "train_negative": neg_count,
            "train_positive": pos_count,
            "scale_pos_weight": scale_pos_weight,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Modell mentve: {model_path}")
    print(f"Metrikak mentve: {output_dir / 'metrics.json'}")
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
