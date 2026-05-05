import argparse
import json
from pathlib import Path

import pandas as pd

from grid_search_xgboost import (
    best_iteration_of,
    best_score_of,
    train_model_with_early_stopping,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Egy konkret XGBoost parameterbeallitas futtatasa early stoppinggal."
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
        default="artifacts/modellfutasok/xgboost_fixed_best_50pct",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--include-partial-csvs", action="store_true")
    parser.add_argument("--sample-csv-ratio", type=float, default=0.5)
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
    parser.add_argument("--max-cat-to-onehot", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="XGBoost futtatas eszkoze. GPU-hoz: --device cuda.",
    )
    return parser.parse_args()


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

    params = {
        "early_stopping_rounds": args.early_stopping_rounds,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "max_cat_to_onehot": args.max_cat_to_onehot,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "gamma": args.gamma,
    }

    print("Modell tanitasa...")
    model = train_model_with_early_stopping(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        params=params,
        seed=args.random_seed,
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

    model_path = output_dir / "xgboost_model.json"
    model.save_model(str(model_path))

    metrics = {
        "params": params,
        "best_iteration": best_iteration_of(model),
        "best_score": best_score_of(model),
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
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Modell mentve: {model_path}")
    print(f"Metrikak mentve: {output_dir / 'metrics.json'}")
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
