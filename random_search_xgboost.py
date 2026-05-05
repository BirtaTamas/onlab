import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from grid_search_xgboost import (
    best_iteration_of,
    best_score_of,
    row_from_metrics,
    train_model_with_early_stopping,
    write_grid_results,
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
        description=(
            "Random search XGBoost regularizacios es mintavetelezesi parameterekre "
            "fix 3 fo parameter mellett."
        )
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
        default="artifacts/modellfutasok/xgboost_random_regularization_earlystop",
        help="Kimeneti mappa a random search eredmenyekhez es a legjobb modellhez.",
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
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="XGBoost futtatas eszkoze. GPU-hoz: --device cuda.",
    )
    parser.add_argument("--max-cat-to-onehot", type=int, default=16)
    return parser.parse_args()


def sample_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_random_params(args: argparse.Namespace, rng: np.random.Generator) -> dict:
    return {
        "early_stopping_rounds": args.early_stopping_rounds,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "max_cat_to_onehot": args.max_cat_to_onehot,
        "min_child_weight": float(rng.choice([5, 8, 10, 12, 15, 20, 25, 30])),
        "subsample": float(rng.uniform(0.60, 0.95)),
        "colsample_bytree": float(rng.uniform(0.60, 0.95)),
        "reg_lambda": sample_log_uniform(rng, 1.0, 12.0),
        "reg_alpha": float(rng.uniform(0.0, 2.0)),
        "gamma": float(rng.uniform(0.0, 5.0)),
    }


def params_run_id(index: int, params: dict) -> str:
    sub = f"{params['subsample']:.2f}".replace(".", "")
    col = f"{params['colsample_bytree']:.2f}".replace(".", "")
    l2 = f"{params['reg_lambda']:.2f}".replace(".", "")
    l1 = f"{params['reg_alpha']:.2f}".replace(".", "")
    gamma = f"{params['gamma']:.2f}".replace(".", "")
    return (
        f"random_{index:03d}"
        f"_mcw{params['min_child_weight']:.0f}"
        f"_sub{sub}"
        f"_col{col}"
        f"_l2{l2}"
        f"_l1{l1}"
        f"_g{gamma}"
    )


def write_random_results(output_dir: Path, rows: List[dict]) -> None:
    results_df = pd.DataFrame(rows).sort_values(
        ["valid_logloss", "valid_roc_auc"],
        ascending=[True, False],
    )
    results_df.to_csv(output_dir / "random_results.csv", index=False)
    (output_dir / "random_results.json").write_text(
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

    rng = np.random.default_rng(args.random_seed)
    rows: List[dict] = []
    best_model = None
    best_params = None
    best_valid_logloss = np.inf

    print(f"Random search meret: {args.n_iter} futas")
    for index in range(1, args.n_iter + 1):
        params = sample_random_params(args, rng)
        run_id = params_run_id(index, params)
        print(f"[{index}/{args.n_iter}] {run_id}")

        model = train_model_with_early_stopping(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            params=params,
            seed=args.random_seed + index,
        )
        train_metrics = evaluate_binary_classifier(model, x_train, y_train)
        valid_metrics = evaluate_binary_classifier(model, x_valid, y_valid)
        row = row_from_metrics(run_id, params, model, train_metrics, valid_metrics)
        row.update(
            {
                "min_child_weight": params["min_child_weight"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "reg_lambda": params["reg_lambda"],
                "reg_alpha": params["reg_alpha"],
                "gamma": params["gamma"],
            }
        )
        rows.append(row)
        write_random_results(output_dir, rows)

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
        "fixed_main_params": {
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "early_stopping_rounds": args.early_stopping_rounds,
            "device": args.device,
        },
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

    print(f"Random search eredmenyek mentve: {output_dir / 'random_results.csv'}")
    print(f"Legjobb modell mentve: {best_model_path}")
    print(json.dumps(best_summary["test"], indent=2))


if __name__ == "__main__":
    main()
