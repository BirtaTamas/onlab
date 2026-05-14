from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_lstm_sequence import (
    build_lstm_model,
    compute_binary_metrics,
    compute_scaler_stats,
    fit_local_ridge_for_round,
    flatten_sequences,
    load_local_round_sequences,
    normalize_csv_path,
    parse_numeric_list,
    probability_fidelity,
    resolve_torch_device,
    write_local_round_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mentett LSTM modell hasznalata egy konkret round lokalis predikciojara."
    )
    parser.add_argument(
        "--lstm-run-dir",
        type=Path,
        default=Path("artifacts/modellfutasok/lstm_sequence_full_cuda_final"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("processed_full"))
    parser.add_argument("--csv", type=str, required=True, help="A vizsgalando CSV path.")
    parser.add_argument("--round-num", type=int, required=True, help="A vizsgalando round_num.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/modellfutasok/lstm_local_round_custom"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0,100.0")
    return parser.parse_args()


def load_train_paths(run_dir: Path, data_root: Path) -> list[Path]:
    manifest_path = run_dir / "sampled_split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Nem talalom a manifestet: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if "split" not in manifest.columns or "csv_path" not in manifest.columns:
        raise ValueError("A manifestben kell split es csv_path oszlop.")
    train_rows = manifest[manifest["split"] == "train"]["csv_path"].tolist()
    if not train_rows:
        raise ValueError("A manifestben nincs train split.")
    return [normalize_csv_path(str(path), data_root=data_root) for path in train_rows]


def load_or_compute_scaler(
    run_dir: Path,
    data_root: Path,
    feature_names: list[str],
    row_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = run_dir / "scaler_stats.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["mean"], cached["std"]

    train_paths = load_train_paths(run_dir, data_root)
    scaler_mean, scaler_std = compute_scaler_stats(
        csv_paths=train_paths,
        feature_names=feature_names,
        row_stride=row_stride,
    )
    np.savez(cache_path, mean=scaler_mean, std=scaler_std)
    print(f"Scaler cache mentve: {cache_path}")
    return scaler_mean, scaler_std


def main() -> None:
    args = parse_args()
    run_dir = args.lstm_run_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    feature_names = list(metrics["features"])
    sequence_length = int(metrics["sequence_length"])
    row_stride = int(metrics.get("sampling", {}).get("row_stride", 1))
    trial = dict(metrics["best_trial"]["params"])
    csv_path = normalize_csv_path(args.csv, data_root=args.data_root)

    scaler_mean, scaler_std = load_or_compute_scaler(
        run_dir=run_dir,
        data_root=args.data_root,
        feature_names=feature_names,
        row_stride=row_stride,
    )

    device = resolve_torch_device(args.device)
    import torch

    model = build_lstm_model((sequence_length, len(feature_names)), trial)
    model.load_state_dict(torch.load(run_dir / "best_lstm_model.pt", map_location=device))
    model.to(device)
    model.eval()

    x_local, y_local, local_df = load_local_round_sequences(
        csv_path=csv_path,
        round_num=args.round_num,
        feature_names=feature_names,
        sequence_length=sequence_length,
        row_stride=row_stride,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )
    if len(local_df) == 0:
        raise ValueError(f"Nem talaltam roundot: csv={csv_path}, round_num={args.round_num}")

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_local).to(device=device, dtype=torch.float32)
        logits = model(x_tensor)
        lstm_prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).astype(float)

    x_flat, flat_feature_names = flatten_sequences(x_local, feature_names)
    ridge_model, ridge_selection = fit_local_ridge_for_round(
        x_round_flat=x_flat,
        y_round_prob=lstm_prob,
        alphas=parse_numeric_list(args.ridge_alphas, float),
    )
    ridge_prob = np.clip(ridge_model.predict(x_flat), 0.0, 1.0)

    local_df = local_df.copy()
    local_df["lstm_probability"] = lstm_prob
    local_df["ridge_probability"] = ridge_prob
    local_df["delta_from_prev"] = local_df["lstm_probability"].diff().fillna(0.0)
    local_df["ridge_delta_from_prev"] = local_df["ridge_probability"].diff().fillna(0.0)

    report_paths = write_local_round_reports(
        output_dir=output_dir,
        round_df=local_df,
        ridge_model=ridge_model,
        flat_feature_names=flat_feature_names,
        x_round_flat=x_flat,
    )

    out_metrics = {
        "source_lstm_run_dir": str(run_dir),
        "csv_path": str(csv_path),
        "round_num": int(args.round_num),
        "row_count": int(len(local_df)),
        "device": str(device),
        "ridge_selection": ridge_selection,
        "fidelity_to_lstm": probability_fidelity(lstm_prob, ridge_prob),
        "lstm_vs_true_labels": compute_binary_metrics(y_local, lstm_prob),
        "ridge_vs_true_labels": compute_binary_metrics(y_local, ridge_prob),
        "reports": report_paths,
    }
    (output_dir / "metrics.json").write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")

    print(f"Lokalis LSTM round mentve: {output_dir / 'local_round_predictions.csv'}")
    print(f"Magyarazat mentve: {output_dir / 'local_round_explainability_summary.md'}")


if __name__ == "__main__":
    main()
