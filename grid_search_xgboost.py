import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from train_xgboost import (
    assign_matches_to_splits,
    build_match_table,
    build_split_manifest,
    collect_csv_metadata,
    evaluate_binary_classifier,
    prepare_datasets,
    read_split_dataframe,
    sample_manifest_rows,
    save_evaluation_plots,
    summarize_manifest_rows,
    summarize_split,
    validate_sampling_args,
    validate_split_ratios,
)


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search a 3 fo XGBoost parameterre early stoppinggal."
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
        default="artifacts/modellfutasok/xgboost_grid_3main_earlystop",
        help="Kimeneti mappa a grid eredmenyekhez es a legjobb modellhez.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--include-partial-csvs", action="store_true")
    parser.add_argument("--sample-csv-ratio", type=float, default=0.3)
    parser.add_argument("--row-stride", type=int, default=1)
    parser.add_argument(
        "--include-categorical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raw place/weapon kategoriak hasznalata. Alapbol kikapcsolva.",
    )
    parser.add_argument(
        "--include-place-categorical",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include-weapon-categorical",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--include-tick", action="store_true")
    parser.add_argument("--drop-utility-features", action="store_true")
    parser.add_argument("--drop-strong-non-utility-features", action="store_true")
    parser.add_argument(
        "--max-depths",
        type=parse_int_list,
        default=parse_int_list("3,4,5"),
        help="Vesszovel elvalasztott max_depth grid, peldaul: 3,4,5.",
    )
    parser.add_argument(
        "--n-estimators-grid",
        type=parse_int_list,
        default=parse_int_list("800,1000,1500"),
        help="Vesszovel elvalasztott n_estimators grid.",
    )
    parser.add_argument(
        "--learning-rates",
        type=parse_float_list,
        default=parse_float_list("0.02,0.03,0.04"),
        help="Vesszovel elvalasztott learning_rate grid.",
    )
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="XGBoost futtatas eszkoze. GPU-hoz: --device cuda.",
    )
    parser.add_argument("--min-child-weight", type=float, default=10.0)
    parser.add_argument("--subsample", type=float, default=0.7)
    parser.add_argument("--colsample-bytree", type=float, default=0.7)
    parser.add_argument("--reg-lambda", type=float, default=4.0)
    parser.add_argument("--reg-alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--max-cat-to-onehot", type=int, default=16)
    return parser.parse_args()


def iter_grid(
    max_depths: Iterable[int],
    n_estimators_grid: Iterable[int],
    learning_rates: Iterable[float],
):
    for max_depth, n_estimators, learning_rate in itertools.product(
        max_depths,
        n_estimators_grid,
        learning_rates,
    ):
        yield {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
        }


def train_model_with_early_stopping(
    x_train,
    y_train,
    x_valid,
    y_valid,
    params: dict,
    seed: int,
):
    import xgboost
    from xgboost import XGBClassifier

    pos_count = max(int((y_train == 1).sum()), 1)
    neg_count = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = neg_count / pos_count
    has_categorical_features = any(str(dtype) == "category" for dtype in x_train.dtypes)

    xgb_major_version = int(xgboost.__version__.split(".", maxsplit=1)[0])
    device = params.get("device", "auto")
    device_kwargs = {}
    if device == "cuda":
        if xgb_major_version >= 2:
            device_kwargs.update(tree_method="hist", device="cuda")
        else:
            device_kwargs.update(tree_method="gpu_hist")
    else:
        device_kwargs.update(tree_method="hist")
        if device == "cpu" and xgb_major_version >= 2:
            device_kwargs.update(device="cpu")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        enable_categorical=has_categorical_features,
        n_jobs=-1,
        early_stopping_rounds=params["early_stopping_rounds"],
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        gamma=params["gamma"],
        max_cat_to_onehot=params["max_cat_to_onehot"],
        **device_kwargs,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=False,
    )
    return model


def best_iteration_of(model) -> int | None:
    value = getattr(model, "best_iteration", None)
    if value is None:
        return None
    return int(value)


def best_score_of(model) -> float | None:
    value = getattr(model, "best_score", None)
    if value is None:
        return None
    return float(value)


def row_from_metrics(run_id: str, params: dict, model, train_metrics: dict, valid_metrics: dict) -> dict:
    return {
        "run_id": run_id,
        "max_depth": params["max_depth"],
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "best_iteration": best_iteration_of(model),
        "best_score": best_score_of(model),
        "train_accuracy": train_metrics["accuracy"],
        "valid_accuracy": valid_metrics["accuracy"],
        "train_logloss": train_metrics["logloss"],
        "valid_logloss": valid_metrics["logloss"],
        "train_roc_auc": train_metrics["roc_auc"],
        "valid_roc_auc": valid_metrics["roc_auc"],
        "train_brier_score": train_metrics["brier_score"],
        "valid_brier_score": valid_metrics["brier_score"],
        "auc_gap": train_metrics["roc_auc"] - valid_metrics["roc_auc"],
    }


def write_grid_results(output_dir: Path, rows: List[dict]) -> None:
    results_df = pd.DataFrame(rows).sort_values(
        ["valid_logloss", "valid_roc_auc"],
        ascending=[True, False],
    )
    results_df.to_csv(output_dir / "grid_results.csv", index=False)
    (output_dir / "grid_results.json").write_text(
        json.dumps(results_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )


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

    split_summaries = [
        summarize_split(split_name, split_matches[split_name])
        for split_name in ["train", "valid", "test"]
    ]
    sampled_split_summaries = [
        summarize_manifest_rows(split_name, sampled_manifest_rows)
        for split_name in ["train", "valid", "test"]
    ]

    pd.DataFrame(manifest_rows).to_csv(output_dir / "split_manifest.csv", index=False)
    pd.DataFrame(sampled_manifest_rows).to_csv(
        output_dir / "sampled_split_manifest.csv",
        index=False,
    )

    train_paths = [Path(row["csv_path"]) for row in sampled_manifest_rows if row["split"] == "train"]
    valid_paths = [Path(row["csv_path"]) for row in sampled_manifest_rows if row["split"] == "valid"]
    test_paths = [Path(row["csv_path"]) for row in sampled_manifest_rows if row["split"] == "test"]

    if not train_paths or not valid_paths or not test_paths:
        raise ValueError("Legalabb egy split ures lett.")

    print("Adatok betoltese...")
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

    fixed_params = {
        "early_stopping_rounds": args.early_stopping_rounds,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "gamma": args.gamma,
        "max_cat_to_onehot": args.max_cat_to_onehot,
        "device": args.device,
    }

    grid = list(
        iter_grid(
            max_depths=args.max_depths,
            n_estimators_grid=args.n_estimators_grid,
            learning_rates=args.learning_rates,
        )
    )
    print(f"Grid meret: {len(grid)} futas")

    rows: List[dict] = []
    best_model = None
    best_params = None
    best_valid_logloss = np.inf

    for index, grid_params in enumerate(grid, start=1):
        params = {**fixed_params, **grid_params}
        run_id = (
            f"d{params['max_depth']}"
            f"_n{params['n_estimators']}"
            f"_lr{str(params['learning_rate']).replace('.', '')}"
        )
        print(f"[{index}/{len(grid)}] {run_id}")

        model = train_model_with_early_stopping(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            params=params,
            seed=args.random_seed,
        )
        train_metrics = evaluate_binary_classifier(model, x_train, y_train)
        valid_metrics = evaluate_binary_classifier(model, x_valid, y_valid)
        rows.append(row_from_metrics(run_id, params, model, train_metrics, valid_metrics))
        write_grid_results(output_dir, rows)

        if valid_metrics["logloss"] < best_valid_logloss:
            best_valid_logloss = valid_metrics["logloss"]
            best_model = model
            best_params = params

    if best_model is None or best_params is None:
        raise RuntimeError("Nem keszult el egyetlen modell sem.")

    split_metrics = {
        "train": evaluate_binary_classifier(best_model, x_train, y_train),
        "valid": evaluate_binary_classifier(best_model, x_valid, y_valid),
        "test": evaluate_binary_classifier(best_model, x_test, y_test),
    }

    plot_artifacts = {}
    for split_name, split_metric_values in split_metrics.items():
        plot_artifacts[split_name] = save_evaluation_plots(
            output_dir=output_dir,
            split_name=split_name,
            split_metrics=split_metric_values,
        )

    best_model_path = output_dir / "best_xgboost_model.json"
    best_model.save_model(str(best_model_path))

    best_summary = {
        "selection_metric": "valid_logloss",
        "best_params": best_params,
        "best_iteration": best_iteration_of(best_model),
        "best_score": best_score_of(best_model),
        "train": split_metrics["train"],
        "valid": split_metrics["valid"],
        "test": split_metrics["test"],
        "plot_artifacts": plot_artifacts,
        "feature_count": len(prepared["feature_names"]),
        "features": prepared["feature_names"],
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
    }
    (output_dir / "best_metrics.json").write_text(
        json.dumps(best_summary, indent=2),
        encoding="utf-8",
    )

    print(f"Grid eredmenyek mentve: {output_dir / 'grid_results.csv'}")
    print(f"Legjobb modell mentve: {best_model_path}")
    print(json.dumps(best_summary["test"], indent=2))


if __name__ == "__main__":
    main()
