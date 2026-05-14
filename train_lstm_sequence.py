import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_xgboost import (
    assign_matches_to_splits,
    build_calibration_curve,
    build_match_table,
    build_split_manifest,
    choose_feature_columns,
    collect_csv_metadata,
    is_utility_column,
    sample_manifest_rows,
    save_evaluation_plots,
    summarize_manifest_rows,
    summarize_split,
    validate_sampling_args,
    validate_split_ratios,
)


STRING_COLUMN_SUFFIXES = (
    "_name",
    "_steamid",
    "_place",
    "_primary_weapon",
    "_secondary_weapon",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idorendi LSTM tanitas CS2 snapshotokra, majd ridge surrogate explainability."
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/modellfutasok/lstm_sequence_distilled",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="",
        help="Ha meg van adva, pontosan ezt a sampled_split_manifest.csv-t hasznalja.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--include-partial-csvs", action="store_true")
    parser.add_argument("--sample-csv-ratio", type=float, default=0.5)
    parser.add_argument("--row-stride", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--window-sample-ratio", type=float, default=1.0)
    parser.add_argument("--include-tick", action="store_true")
    parser.add_argument("--drop-utility-features", action="store_true")
    parser.add_argument("--drop-strong-non-utility-features", action="store_true")
    parser.add_argument("--batch-sizes", type=str, default="256,512")
    parser.add_argument("--lstm-units", type=str, default="64,96")
    parser.add_argument("--dense-units", type=str, default="0,32")
    parser.add_argument("--dropouts", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--learning-rates", type=str, default="0.001,0.0005")
    parser.add_argument("--max-trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch eszkoz. auto: CUDA, ha elerheto, kulonben CPU.",
    )
    parser.add_argument(
        "--run-eagerly",
        action="store_true",
        help="Keras eager mod debug/stabilitasi futtatasra. Lassabb lehet, de segithet az elso batch beragadasanal.",
    )
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0,100.0")
    parser.add_argument(
        "--local-explain-split",
        choices=["train", "valid", "test"],
        default="test",
        help="Melyik splitbol valasszunk roundot a lokalis ridge magyarazathoz.",
    )
    parser.add_argument(
        "--local-explain-csv",
        type=str,
        default="",
        help="Opcionális konkret CSV path a manifestbol a lokalis round-magyarazathoz.",
    )
    parser.add_argument(
        "--local-explain-round-num",
        type=int,
        default=None,
        help="Opcionális round sorszam a lokalis magyarazathoz.",
    )
    return parser.parse_args()


def parse_numeric_list(raw: str, cast_type):
    values = []
    for item in raw.split(","):
        text = item.strip()
        if text:
            values.append(cast_type(text))
    if not values:
        raise ValueError("Legalabb egy erteket meg kell adni a listaban.")
    return values


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def resolve_torch_device(requested_device: str):
    import torch

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A --device cuda lett megadva, de a PyTorch nem lat CUDA GPU-t. "
                "Ellenorizd az NVIDIA drivert es a CUDA-s PyTorch telepitest."
            )
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_torch_device(device) -> Dict[str, object]:
    import torch

    info = {
        "requested": str(device),
        "selected": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": None,
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
    return info


def load_manifest_from_path(manifest_path: Path) -> List[Dict[str, object]]:
    manifest_df = pd.read_csv(manifest_path)
    required = {"split", "csv_path"}
    missing = required - set(manifest_df.columns)
    if missing:
        raise ValueError(f"Hianyzik a manifestbol: {sorted(missing)}")
    return manifest_df.to_dict(orient="records")


def normalize_csv_path(raw_path: str, data_root: Path) -> Path:
    normalized = str(raw_path).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.exists():
        return candidate

    if candidate.is_absolute():
        return candidate

    project_root = Path.cwd()
    project_candidate = project_root / candidate
    if project_candidate.exists():
        return project_candidate

    data_root_name = data_root.name
    parts = candidate.parts
    if parts and parts[0] == data_root_name:
        root_candidate = data_root.parent / candidate
        if root_candidate.exists():
            return root_candidate

    root_join_candidate = data_root / candidate
    if root_join_candidate.exists():
        return root_join_candidate

    return candidate


def read_csv_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def infer_lstm_schema_overrides(csv_path: Path) -> Dict[str, pl.DataType]:
    return {
        col_name: pl.Utf8
        for col_name in read_csv_header(csv_path)
        if col_name.endswith(STRING_COLUMN_SUFFIXES)
    }


def build_or_load_manifest(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict], List[Dict]]:
    data_root = Path(args.data_root)
    if args.manifest_path:
        sampled_manifest_rows = load_manifest_from_path(Path(args.manifest_path))
        split_summaries = []
        sampled_split_summaries = [
            summarize_manifest_rows(split_name, sampled_manifest_rows)
            for split_name in ["train", "valid", "test"]
        ]
        return sampled_manifest_rows, split_summaries, sampled_split_summaries

    metadata = collect_csv_metadata(data_root, include_partial_csvs=args.include_partial_csvs)
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
    return sampled_manifest_rows, split_summaries, sampled_split_summaries


def read_single_csv(csv_path: Path, row_stride: int) -> pl.DataFrame:
    df = pl.read_csv(
        str(csv_path),
        schema_overrides=infer_lstm_schema_overrides(csv_path),
        infer_schema_length=10000,
    )
    if row_stride > 1:
        df = (
            df.with_row_index("__row_idx")
            .filter((pl.col("__row_idx") % row_stride) == 0)
            .drop("__row_idx")
        )
    return df


def materialize_numeric_features(
    df: pl.DataFrame,
    feature_names: Sequence[str],
    label_col: str = "ct_win",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "round_num" not in df.columns:
        raise ValueError("A temporal sequence epiteshez kell a round_num oszlop.")

    existing_feature_names = [feature_name for feature_name in feature_names if feature_name in df.columns]
    required_columns = [col_name for col_name in [label_col, "round_num"] if col_name in df.columns]
    selected_columns = list(dict.fromkeys(existing_feature_names + required_columns))
    pandas_df = df.select(selected_columns).to_pandas()
    feature_df = (
        pandas_df.reindex(columns=list(feature_names), fill_value=0.0)
        .copy()
    )

    for col_name in feature_df.columns:
        if pd.api.types.is_bool_dtype(feature_df[col_name]):
            feature_df[col_name] = feature_df[col_name].astype(np.int8)
        feature_df[col_name] = feature_df[col_name].fillna(0)
    x = feature_df.to_numpy(dtype=np.float32)
    y = pandas_df[label_col].fillna(0).astype(np.int64).to_numpy()
    round_ids = pandas_df["round_num"].fillna(-1).astype(np.int64).to_numpy()
    return x, y, round_ids


def extract_sequence_metadata(df: pl.DataFrame) -> pd.DataFrame:
    selected_columns = ["round_num"]
    optional_columns = ["tick", "seconds_in_round", "bomb_planted"]
    for col_name in optional_columns:
        if col_name in df.columns:
            selected_columns.append(col_name)
    metadata_df = df.select(selected_columns).to_pandas()
    if "tick" not in metadata_df.columns:
        metadata_df["tick"] = np.arange(len(metadata_df), dtype=np.int64)
    if "seconds_in_round" not in metadata_df.columns:
        metadata_df["seconds_in_round"] = np.nan
    if "bomb_planted" not in metadata_df.columns:
        metadata_df["bomb_planted"] = 0
    metadata_df["round_num"] = metadata_df["round_num"].fillna(-1).astype(np.int64)
    metadata_df["tick"] = metadata_df["tick"].fillna(-1).astype(np.int64)
    metadata_df["bomb_planted"] = metadata_df["bomb_planted"].fillna(0).astype(np.int64)
    return metadata_df


def build_sequences_for_file(
    x_rows: np.ndarray,
    y_rows: np.ndarray,
    round_ids: np.ndarray,
    sequence_length: int,
    metadata_df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequences: List[np.ndarray] = []
    labels: List[int] = []
    metadata_rows: List[Dict[str, object]] = []
    unique_rounds = pd.unique(round_ids)
    feature_dim = x_rows.shape[1]

    for round_id in unique_rounds:
        idxs = np.flatnonzero(round_ids == round_id)
        if idxs.size == 0:
            continue
        round_x = x_rows[idxs]
        round_y = y_rows[idxs]
        for end_idx in range(len(idxs)):
            start_idx = max(0, end_idx - sequence_length + 1)
            window = round_x[start_idx : end_idx + 1]
            if window.shape[0] < sequence_length:
                pad = np.zeros((sequence_length - window.shape[0], feature_dim), dtype=np.float32)
                window = np.vstack([pad, window])
            sequences.append(window.astype(np.float32, copy=False))
            labels.append(int(round_y[end_idx]))
            if metadata_df is not None:
                row_meta = metadata_df.iloc[idxs[end_idx]]
                metadata_rows.append(
                    {
                        "csv_path": str(csv_path) if csv_path is not None else "",
                        "round_num": int(round_id),
                        "tick": int(row_meta["tick"]),
                        "seconds_in_round": (
                            None if pd.isna(row_meta["seconds_in_round"]) else float(row_meta["seconds_in_round"])
                        ),
                        "bomb_planted": int(row_meta["bomb_planted"]),
                    }
                )

    if not sequences:
        return (
            np.zeros((0, sequence_length, feature_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            pd.DataFrame(columns=["csv_path", "round_num", "tick", "seconds_in_round", "bomb_planted"]),
        )

    metadata_out = pd.DataFrame(metadata_rows)
    return np.stack(sequences), np.asarray(labels, dtype=np.int64), metadata_out


def maybe_subsample_sequences(
    x_seq: np.ndarray,
    y_seq: np.ndarray,
    ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if ratio >= 1.0 or len(y_seq) == 0:
        return x_seq, y_seq
    keep_count = max(1, int(round(len(y_seq) * ratio)))
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(y_seq), size=keep_count, replace=False))
    return x_seq[selected], y_seq[selected]


def build_split_sequences(
    csv_paths: Sequence[Path],
    feature_names: Sequence[str],
    sequence_length: int,
    row_stride: int,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    sample_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    all_sequences: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_metadata: List[pd.DataFrame] = []

    for csv_path in csv_paths:
        df = read_single_csv(csv_path, row_stride=row_stride)
        x_rows, y_rows, round_ids = materialize_numeric_features(df, feature_names=feature_names)
        row_metadata = extract_sequence_metadata(df)
        x_rows = (x_rows - scaler_mean) / scaler_std
        x_seq, y_seq, seq_metadata = build_sequences_for_file(
            x_rows=x_rows,
            y_rows=y_rows,
            round_ids=round_ids,
            sequence_length=sequence_length,
            metadata_df=row_metadata,
            csv_path=csv_path,
        )
        if len(y_seq) == 0:
            continue
        x_seq, y_seq = maybe_subsample_sequences(
            x_seq,
            y_seq,
            ratio=sample_ratio,
            seed=seed + len(all_sequences),
        )
        if len(seq_metadata) != len(y_seq):
            if len(y_seq) == 0:
                continue
            keep_count = len(y_seq)
            rng = np.random.default_rng(seed + len(all_sequences))
            selected = np.sort(rng.choice(len(seq_metadata), size=keep_count, replace=False))
            seq_metadata = seq_metadata.iloc[selected].reset_index(drop=True)
        all_sequences.append(x_seq)
        all_labels.append(y_seq)
        all_metadata.append(seq_metadata)

    if not all_sequences:
        raise ValueError("Nem sikerult sequence mintakat generalni a splithez.")

    x_full = np.concatenate(all_sequences, axis=0)
    y_full = np.concatenate(all_labels, axis=0)
    metadata_full = pd.concat(all_metadata, ignore_index=True)
    return x_full, y_full, metadata_full.reset_index(drop=True)


def load_sequences_from_csv(
    csv_path: Path,
    feature_names: Sequence[str],
    sequence_length: int,
    row_stride: int,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    sample_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = read_single_csv(csv_path, row_stride=row_stride)
    x_rows, y_rows, round_ids = materialize_numeric_features(df, feature_names=feature_names)
    row_metadata = extract_sequence_metadata(df)
    x_rows = (x_rows - scaler_mean) / scaler_std
    x_seq, y_seq, seq_metadata = build_sequences_for_file(
        x_rows=x_rows,
        y_rows=y_rows,
        round_ids=round_ids,
        sequence_length=sequence_length,
        metadata_df=row_metadata,
        csv_path=csv_path,
    )
    if len(y_seq) == 0:
        return x_seq, y_seq, seq_metadata
    x_seq, y_seq = maybe_subsample_sequences(
        x_seq,
        y_seq,
        ratio=sample_ratio,
        seed=seed,
    )
    if len(seq_metadata) != len(y_seq):
        keep_count = len(y_seq)
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(seq_metadata), size=keep_count, replace=False))
        seq_metadata = seq_metadata.iloc[selected].reset_index(drop=True)
    else:
        seq_metadata = seq_metadata.reset_index(drop=True)
    return x_seq, y_seq, seq_metadata


def build_feature_names_from_headers(
    csv_paths: Sequence[Path],
    include_tick: bool,
    drop_utility_features: bool,
    drop_strong_non_utility_features: bool,
) -> Tuple[List[str], List[str]]:
    union_columns: Dict[str, pl.DataType] = {"ct_win": pl.Int8}
    for csv_path in csv_paths:
        schema_overrides = infer_lstm_schema_overrides(csv_path)
        sample_df = pl.read_csv(
            str(csv_path),
            n_rows=256,
            schema_overrides=schema_overrides,
            infer_schema_length=1000,
        )
        for col_name, dtype in zip(sample_df.columns, sample_df.dtypes):
            previous = union_columns.get(col_name)
            if previous is None:
                union_columns[col_name] = dtype
            elif previous != pl.Utf8 and dtype == pl.Utf8:
                union_columns[col_name] = dtype
    schema_df = pl.DataFrame(schema=union_columns)
    feature_plan = choose_feature_columns(
        schema_df,
        label_col="ct_win",
        include_categorical=False,
        include_place_categorical=False,
        include_weapon_categorical=False,
        include_tick=include_tick,
        drop_utility_features=drop_utility_features,
        drop_strong_non_utility_features=drop_strong_non_utility_features,
    )
    feature_names = feature_plan["numeric"]
    if "round_num" not in feature_names:
        feature_names = ["round_num"] + feature_names
    return feature_names, feature_plan["dropped"]


def compute_scaler_stats(
    csv_paths: Sequence[Path],
    feature_names: Sequence[str],
    row_stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    log_progress(f"Scaler stats inditasa, train CSV-k: {len(csv_paths)}")
    total_count = 0
    sum_vector = None
    sumsq_vector = None

    for idx, csv_path in enumerate(csv_paths, start=1):
        if idx == 1 or idx % 25 == 0 or idx == len(csv_paths):
            log_progress(f"Scaler CSV {idx}/{len(csv_paths)}: {csv_path.name}")
        df = read_single_csv(csv_path, row_stride=row_stride)
        x_rows, _, _ = materialize_numeric_features(df, feature_names=feature_names)
        if x_rows.size == 0:
            continue
        x64 = x_rows.astype(np.float64, copy=False)
        batch_sum = x64.sum(axis=0)
        batch_sumsq = np.square(x64).sum(axis=0)
        if sum_vector is None:
            sum_vector = batch_sum
            sumsq_vector = batch_sumsq
        else:
            sum_vector += batch_sum
            sumsq_vector += batch_sumsq
        total_count += x64.shape[0]

    if total_count == 0 or sum_vector is None or sumsq_vector is None:
        raise ValueError("Nem sikerult scaler statisztikat szamolni a train splitbol.")

    mean = sum_vector / total_count
    variance = np.maximum((sumsq_vector / total_count) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std == 0.0] = 1.0
    log_progress(f"Scaler stats kesz, osszes train sor: {total_count}")
    return mean.astype(np.float32), std.astype(np.float32)


def build_stream_file_infos(
    csv_paths: Sequence[Path],
    row_stride: int,
    sample_ratio: float,
    seed: int,
) -> List[Dict[str, object]]:
    log_progress(f"Streaming file infok epitese, CSV-k: {len(csv_paths)}")
    file_infos: List[Dict[str, object]] = []
    for idx, csv_path in enumerate(csv_paths, start=1):
        if idx == 1 or idx % 25 == 0 or idx == len(csv_paths):
            log_progress(f"File info CSV {idx}/{len(csv_paths)}: {csv_path.name}")
        row_count = int(read_single_csv(csv_path, row_stride=row_stride).height)
        if row_count <= 0:
            continue
        sampled_count = row_count if sample_ratio >= 1.0 else max(1, int(round(row_count * sample_ratio)))
        file_infos.append(
            {
                "csv_path": csv_path,
                "row_count": row_count,
                "sampled_count": sampled_count,
                "seed": seed + idx,
            }
        )
    if not file_infos:
        raise ValueError("Nem maradt egyetlen hasznalhato CSV sem a splitben.")
    log_progress(
        f"Streaming file infok kesz, hasznalhato CSV-k: {len(file_infos)}, "
        f"becsult sequence-ek: {sum(int(info['sampled_count']) for info in file_infos)}"
    )
    return file_infos


class StreamingSequence:
    def __init__(
        self,
        file_infos: Sequence[Dict[str, object]],
        feature_names: Sequence[str],
        sequence_length: int,
        row_stride: int,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        batch_size: int,
    ) -> None:
        self.file_infos = list(file_infos)
        self.feature_names = list(feature_names)
        self.sequence_length = int(sequence_length)
        self.row_stride = int(row_stride)
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.batch_size = int(batch_size)
        self.total_sequences = int(sum(int(info["sampled_count"]) for info in self.file_infos))
        self.cumulative_counts = np.cumsum([int(info["sampled_count"]) for info in self.file_infos])
        self._cache_file_idx = None
        self._cache_x = None
        self._cache_y = None

    def __len__(self) -> int:
        return int(math.ceil(self.total_sequences / self.batch_size))

    def _load_file_cache(self, file_idx: int) -> None:
        if self._cache_file_idx == file_idx:
            return
        info = self.file_infos[file_idx]
        log_progress(
            f"Batch cache betoltes: file {file_idx + 1}/{len(self.file_infos)} - "
            f"{Path(info['csv_path']).name}"
        )
        sample_ratio = float(info["sampled_count"]) / max(int(info["row_count"]), 1)
        x_seq, y_seq, _ = load_sequences_from_csv(
            csv_path=Path(info["csv_path"]),
            feature_names=self.feature_names,
            sequence_length=self.sequence_length,
            row_stride=self.row_stride,
            scaler_mean=self.scaler_mean,
            scaler_std=self.scaler_std,
            sample_ratio=sample_ratio,
            seed=int(info["seed"]),
        )
        self._cache_file_idx = file_idx
        self._cache_x = x_seq
        self._cache_y = y_seq
        log_progress(
            f"Batch cache kesz: {Path(info['csv_path']).name}, sequence-ek: {len(y_seq)}"
        )

    def __getitem__(self, batch_idx: int):
        if batch_idx == 0 or (batch_idx + 1) % 100 == 0:
            log_progress(f"Sequence __getitem__ indul: batch {batch_idx + 1}/{len(self)}")
        start = batch_idx * self.batch_size
        end = min((batch_idx + 1) * self.batch_size, self.total_sequences)
        if start >= end:
            raise IndexError(batch_idx)

        batch_x_parts: List[np.ndarray] = []
        batch_y_parts: List[np.ndarray] = []
        cursor = start

        while cursor < end:
            file_idx = int(np.searchsorted(self.cumulative_counts, cursor, side="right"))
            prev_total = 0 if file_idx == 0 else int(self.cumulative_counts[file_idx - 1])
            offset = cursor - prev_total
            file_end = int(self.cumulative_counts[file_idx])
            take = min(end - cursor, file_end - cursor)
            self._load_file_cache(file_idx)
            batch_x_parts.append(self._cache_x[offset : offset + take])
            batch_y_parts.append(self._cache_y[offset : offset + take])
            cursor += take

        batch_x = np.concatenate(batch_x_parts, axis=0)
        batch_y = np.concatenate(batch_y_parts, axis=0)
        if batch_idx == 0 or (batch_idx + 1) % 100 == 0:
            log_progress(
                f"Sequence __getitem__ kesz: batch {batch_idx + 1}/{len(self)}, "
                f"shape={batch_x.shape}"
            )
        return batch_x, batch_y


def make_keras_sequence_class():
    import tensorflow as tf

    class KerasStreamingSequence(tf.keras.utils.Sequence):
        def __init__(
            self,
            file_infos: Sequence[Dict[str, object]],
            feature_names: Sequence[str],
            sequence_length: int,
            row_stride: int,
            scaler_mean: np.ndarray,
            scaler_std: np.ndarray,
            batch_size: int,
        ) -> None:
            super().__init__()
            self._impl = StreamingSequence(
                file_infos=file_infos,
                feature_names=feature_names,
                sequence_length=sequence_length,
                row_stride=row_stride,
                scaler_mean=scaler_mean,
                scaler_std=scaler_std,
                batch_size=batch_size,
            )

        def __len__(self) -> int:
            return len(self._impl)

        def __getitem__(self, index: int):
            return self._impl[index]

    return KerasStreamingSequence


def make_progress_callback_class():
    import tensorflow as tf

    class KerasProgressLoggingCallback(tf.keras.callbacks.Callback):
        def __init__(self, total_train_batches: int, total_valid_batches: int) -> None:
            super().__init__()
            self.total_train_batches = int(total_train_batches)
            self.total_valid_batches = int(total_valid_batches)

        def on_epoch_begin(self, epoch, logs=None) -> None:
            log_progress(
                f"Epoch {epoch + 1} indul, train_batches={self.total_train_batches}, "
                f"valid_batches={self.total_valid_batches}"
            )

        def on_train_batch_end(self, batch, logs=None) -> None:
            batch_index = int(batch) + 1
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == self.total_train_batches:
                loss = None if logs is None else logs.get("loss")
                if loss is None:
                    log_progress(f"Train batch {batch_index}/{self.total_train_batches}")
                else:
                    log_progress(
                        f"Train batch {batch_index}/{self.total_train_batches}, loss={float(loss):.5f}"
                    )

        def on_train_batch_begin(self, batch, logs=None) -> None:
            batch_index = int(batch) + 1
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == self.total_train_batches:
                log_progress(f"Train batch begin {batch_index}/{self.total_train_batches}")

        def on_epoch_end(self, epoch, logs=None) -> None:
            logs = logs or {}
            loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            pieces = [f"Epoch {epoch + 1} kesz"]
            if loss is not None:
                pieces.append(f"loss={float(loss):.5f}")
            if val_loss is not None:
                pieces.append(f"val_loss={float(val_loss):.5f}")
            log_progress(", ".join(pieces))

    return KerasProgressLoggingCallback


def predict_split_streaming(
    model,
    file_infos: Sequence[Dict[str, object]],
    feature_names: Sequence[str],
    sequence_length: int,
    row_stride: int,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    log_progress(f"Predikcio inditasa splitre, CSV-k: {len(file_infos)}")
    all_probs: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_metadata: List[pd.DataFrame] = []
    for idx, info in enumerate(file_infos, start=1):
        if idx == 1 or idx % 25 == 0 or idx == len(file_infos):
            log_progress(f"Predikcio CSV {idx}/{len(file_infos)}: {Path(info['csv_path']).name}")
        sample_ratio = float(info["sampled_count"]) / max(int(info["row_count"]), 1)
        x_seq, y_seq, metadata_df = load_sequences_from_csv(
            csv_path=Path(info["csv_path"]),
            feature_names=feature_names,
            sequence_length=sequence_length,
            row_stride=row_stride,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            sample_ratio=sample_ratio,
            seed=int(info["seed"]),
        )
        if len(y_seq) == 0:
            continue
        probs = model.predict(x_seq, verbose=0).reshape(-1)
        all_probs.append(probs)
        all_y.append(y_seq)
        all_metadata.append(metadata_df)
    if not all_probs:
        raise ValueError("Nem sikerult predikciot kesziteni a splithez.")
    log_progress(f"Predikcio kesz, osszes sequence: {sum(len(y) for y in all_y)}")
    return np.concatenate(all_probs), np.concatenate(all_y), pd.concat(all_metadata, ignore_index=True)


def load_local_round_sequences(
    csv_path: Path,
    round_num: int,
    feature_names: Sequence[str],
    sequence_length: int,
    row_stride: int,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    selected_ticks: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_seq, y_seq, metadata_df = load_sequences_from_csv(
        csv_path=csv_path,
        feature_names=feature_names,
        sequence_length=sequence_length,
        row_stride=row_stride,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        sample_ratio=1.0,
        seed=0,
    )
    mask = (metadata_df["round_num"] == int(round_num)).to_numpy()
    round_x = x_seq[mask]
    round_y = y_seq[mask]
    round_meta = metadata_df.loc[mask].reset_index(drop=True)
    if selected_ticks is not None:
        tick_set = {int(tick) for tick in selected_ticks}
        tick_mask = round_meta["tick"].astype(int).isin(tick_set).to_numpy()
        round_x = round_x[tick_mask]
        round_y = round_y[tick_mask]
        round_meta = round_meta.loc[tick_mask].reset_index(drop=True)
    return round_x, round_y, round_meta


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
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
        "calibration_curve": build_calibration_curve(y_true=np.asarray(y_true), y_prob=np.asarray(y_prob)),
    }
    unique_labels = np.unique(y_true)
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(unique_labels) == 2 else None
    return metrics


def probability_fidelity(y_teacher: np.ndarray, y_student: np.ndarray) -> Dict[str, float]:
    teacher = np.asarray(y_teacher, dtype=np.float64)
    student = np.asarray(y_student, dtype=np.float64)
    corr = float(np.corrcoef(teacher, student)[0, 1]) if len(teacher) > 1 else 0.0
    return {
        "mae": float(mean_absolute_error(teacher, student)),
        "rmse": float(math.sqrt(mean_squared_error(teacher, student))),
        "r2": float(r2_score(teacher, student)),
        "pearson_corr": corr,
    }


def make_trial_grid(args: argparse.Namespace) -> List[Dict[str, float]]:
    grid = list(
        itertools.product(
            parse_numeric_list(args.batch_sizes, int),
            parse_numeric_list(args.lstm_units, int),
            parse_numeric_list(args.dense_units, int),
            parse_numeric_list(args.dropouts, float),
            parse_numeric_list(args.learning_rates, float),
        )
    )
    rng = np.random.default_rng(args.random_seed)
    rng.shuffle(grid)
    selected = grid[: min(args.max_trials, len(grid))]
    return [
        {
            "batch_size": int(batch_size),
            "lstm_units": int(lstm_units),
            "dense_units": int(dense_units),
            "dropout": float(dropout),
            "learning_rate": float(learning_rate),
        }
        for batch_size, lstm_units, dense_units, dropout, learning_rate in selected
    ]


def build_lstm_model(
    input_shape: Tuple[int, int],
    trial: Dict[str, float],
):
    import torch
    from torch import nn

    _, feature_dim = input_shape

    class LSTMSequenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=trial["lstm_units"],
                batch_first=True,
            )
            self.dropout = nn.Dropout(trial["dropout"]) if trial["dropout"] > 0.0 else nn.Identity()
            if trial["dense_units"] > 0:
                self.hidden = nn.Linear(trial["lstm_units"], trial["dense_units"])
                self.hidden_activation = nn.ReLU()
                self.output = nn.Linear(trial["dense_units"], 1)
            else:
                self.hidden = None
                self.hidden_activation = None
                self.output = nn.Linear(trial["lstm_units"], 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            x_last = lstm_out[:, -1, :]
            x_last = self.dropout(x_last)
            if self.hidden is not None:
                x_last = self.hidden_activation(self.hidden(x_last))
                x_last = self.dropout(x_last)
            logits = self.output(x_last).squeeze(-1)
            return logits

    return LSTMSequenceModel()


def train_torch_lstm_model(
    model,
    train_sequence,
    valid_sequence,
    trial: Dict[str, float],
    epochs: int,
    patience: int,
    device,
) -> Tuple[object, Dict[str, List[float]]]:
    import copy
    import torch

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=trial["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    history = {"loss": [], "val_loss": []}
    best_state = None
    best_val_loss = None
    epochs_without_improvement = 0

    train_batches = len(train_sequence)
    valid_batches = len(valid_sequence)

    for epoch in range(epochs):
        log_progress(
            f"Epoch {epoch + 1} indul, train_batches={train_batches}, valid_batches={valid_batches}"
        )
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0

        for batch_idx in range(train_batches):
            if batch_idx == 0 or (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == train_batches:
                log_progress(f"Train batch begin {batch_idx + 1}/{train_batches}")
            x_np, y_np = train_sequence[batch_idx]
            x_tensor = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            y_tensor = torch.from_numpy(y_np.astype(np.float32, copy=False)).to(device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()

            batch_size = int(y_np.shape[0])
            train_loss_sum += float(loss.detach().cpu().item()) * batch_size
            train_weight_sum += batch_size
            if batch_idx == 0 or (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == train_batches:
                log_progress(
                    f"Train batch {batch_idx + 1}/{train_batches}, loss={float(loss.detach().cpu().item()):.5f}"
                )

        model.eval()
        valid_loss_sum = 0.0
        valid_weight_sum = 0
        with torch.no_grad():
            for batch_idx in range(valid_batches):
                x_np, y_np = valid_sequence[batch_idx]
                x_tensor = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
                y_tensor = torch.from_numpy(y_np.astype(np.float32, copy=False)).to(device=device)
                logits = model(x_tensor)
                loss = criterion(logits, y_tensor)
                batch_size = int(y_np.shape[0])
                valid_loss_sum += float(loss.detach().cpu().item()) * batch_size
                valid_weight_sum += batch_size

        train_loss = train_loss_sum / max(train_weight_sum, 1)
        val_loss = valid_loss_sum / max(valid_weight_sum, 1)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        log_progress(f"Epoch {epoch + 1} kesz, loss={train_loss:.5f}, val_loss={val_loss:.5f}")

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_split_streaming_torch(
    model,
    file_infos: Sequence[Dict[str, object]],
    feature_names: Sequence[str],
    sequence_length: int,
    row_stride: int,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    device,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    import torch

    model = model.to(device)
    model.eval()

    log_progress(f"Predikcio inditasa splitre, CSV-k: {len(file_infos)}")
    all_probs: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_metadata: List[pd.DataFrame] = []
    with torch.no_grad():
        for idx, info in enumerate(file_infos, start=1):
            if idx == 1 or idx % 25 == 0 or idx == len(file_infos):
                log_progress(f"Predikcio CSV {idx}/{len(file_infos)}: {Path(info['csv_path']).name}")
            sample_ratio = float(info["sampled_count"]) / max(int(info["row_count"]), 1)
            x_seq, y_seq, metadata_df = load_sequences_from_csv(
                csv_path=Path(info["csv_path"]),
                feature_names=feature_names,
                sequence_length=sequence_length,
                row_stride=row_stride,
                scaler_mean=scaler_mean,
                scaler_std=scaler_std,
                sample_ratio=sample_ratio,
                seed=int(info["seed"]),
            )
            if len(y_seq) == 0:
                continue
            x_tensor = torch.from_numpy(x_seq).to(device=device, dtype=torch.float32)
            logits = model(x_tensor)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            all_probs.append(probs)
            all_y.append(y_seq)
            all_metadata.append(metadata_df)
    if not all_probs:
        raise ValueError("Nem sikerult predikciot kesziteni a splithez.")
    log_progress(f"Predikcio kesz, osszes sequence: {sum(len(y) for y in all_y)}")
    return np.concatenate(all_probs), np.concatenate(all_y), pd.concat(all_metadata, ignore_index=True)


def flatten_sequences(x_seq: np.ndarray, feature_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    seq_len = x_seq.shape[1]
    flat_names: List[str] = []
    for lag in range(seq_len):
        lag_from_now = seq_len - lag - 1
        for feature_name in feature_names:
            flat_names.append(f"lag_{lag_from_now:02d}__{feature_name}")
    return x_seq.reshape(x_seq.shape[0], -1), flat_names


def pick_local_round(
    metadata_df: pd.DataFrame,
    probabilities: np.ndarray,
    split_name: str,
    requested_csv: str,
    requested_round_num: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    local_df = metadata_df.copy()
    local_df["lstm_probability"] = np.asarray(probabilities, dtype=np.float64)

    if requested_csv:
        local_df = local_df[local_df["csv_path"] == requested_csv].copy()
        if local_df.empty:
            raise ValueError(f"Nem talaltam ilyen CSV-t a {split_name} splitben: {requested_csv}")

    if requested_round_num is not None:
        local_df = local_df[local_df["round_num"] == int(requested_round_num)].copy()
        if local_df.empty:
            raise ValueError(
                f"Nem talaltam ilyen roundot a {split_name} splitben: round_num={requested_round_num}"
            )

    if requested_csv or requested_round_num is not None:
        local_df = local_df.sort_values(["tick"]).reset_index(drop=True)
        chosen = {
            "selection_mode": "user_selected",
            "csv_path": str(local_df["csv_path"].iloc[0]),
            "round_num": int(local_df["round_num"].iloc[0]),
            "split": split_name,
        }
        return local_df, chosen

    grouped_rounds = []
    for (csv_path, round_num), group in local_df.groupby(["csv_path", "round_num"], sort=False):
        ordered = group.sort_values("tick").reset_index(drop=True)
        deltas = ordered["lstm_probability"].diff().fillna(0.0)
        score = float(deltas.abs().max())
        grouped_rounds.append(
            {
                "csv_path": str(csv_path),
                "round_num": int(round_num),
                "score": score,
                "row_count": int(len(ordered)),
            }
        )
    if not grouped_rounds:
        raise ValueError("Nem sikerult roundot valasztani a lokalis magyarazathoz.")
    picked = sorted(grouped_rounds, key=lambda row: (row["score"], row["row_count"]), reverse=True)[0]
    chosen_df = local_df[
        (local_df["csv_path"] == picked["csv_path"]) & (local_df["round_num"] == picked["round_num"])
    ].copy()
    chosen_df = chosen_df.sort_values("tick").reset_index(drop=True)
    picked["selection_mode"] = "auto_max_probability_jump"
    picked["split"] = split_name
    return chosen_df, picked


def fit_local_ridge_for_round(
    x_round_flat: np.ndarray,
    y_round_prob: np.ndarray,
    alphas: Iterable[float],
) -> Tuple[Ridge, Dict[str, object]]:
    alpha_values = [float(alpha) for alpha in alphas]
    sample_count = len(y_round_prob)
    if sample_count < 3:
        model = Ridge(alpha=alpha_values[0], random_state=42)
        model.fit(x_round_flat, y_round_prob)
        return model, {"alpha": float(alpha_values[0]), "cv_folds": 1}

    cv_folds = min(5, sample_count)
    ridge_cv = RidgeCV(alphas=alpha_values, cv=cv_folds)
    ridge_cv.fit(x_round_flat, y_round_prob)
    model = Ridge(alpha=float(ridge_cv.alpha_), random_state=42)
    model.fit(x_round_flat, y_round_prob)
    return model, {"alpha": float(ridge_cv.alpha_), "cv_folds": int(cv_folds)}


def write_local_round_reports(
    output_dir: Path,
    round_df: pd.DataFrame,
    ridge_model: Ridge,
    flat_feature_names: Sequence[str],
    x_round_flat: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    coef_df = pd.DataFrame(
        {
            "feature": list(flat_feature_names),
            "coefficient": ridge_model.coef_.astype(float),
            "abs_coefficient": np.abs(ridge_model.coef_.astype(float)),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coef_df["base_feature"] = coef_df["feature"].str.replace(r"^lag_\d+__", "", regex=True)
    coef_df["is_utility"] = coef_df["base_feature"].map(is_utility_column)
    coef_path = output_dir / "local_round_ridge_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    top_jumps = round_df.reindex(
        round_df["delta_from_prev"].abs().sort_values(ascending=False).index
    ).head(10)
    summary_lines = [
        "# Local Round Explainability",
        "",
        f"- csv_path: `{round_df['csv_path'].iloc[0]}`",
        f"- round_num: `{int(round_df['round_num'].iloc[0])}`",
        "",
        "## Largest probability jumps",
        "",
    ]
    for _, row in top_jumps.iterrows():
        seconds_text = "NA" if pd.isna(row["seconds_in_round"]) else f"{float(row['seconds_in_round']):.2f}"
        summary_lines.append(
            f"- tick `{int(row['tick'])}`, seconds `{seconds_text}`, "
            f"LSTM `{float(row['lstm_probability']):.4f}`, delta `{float(row['delta_from_prev']):+.4f}`"
        )
    summary_lines.extend(["", "## Top 15 local ridge features", ""])
    for _, row in coef_df.head(15).iterrows():
        summary_lines.append(
            f"- `{row['feature']}`: coefficient `{row['coefficient']:.6f}`, |coef| `{row['abs_coefficient']:.6f}`"
        )
    summary_lines.extend(["", "## Top 10 utility ridge features", ""])
    utility_top = coef_df[coef_df["is_utility"]].head(10)
    if utility_top.empty:
        summary_lines.append("- No utility features in the local top list.")
    else:
        for _, row in utility_top.iterrows():
            direction = "raises CT win probability" if row["coefficient"] > 0 else "lowers CT win probability"
            summary_lines.append(
                f"- `{row['feature']}`: coefficient `{row['coefficient']:.6f}` ({direction})"
            )
    summary_lines.extend(["", "## Top 10 non-utility ridge features", ""])
    non_utility_top = coef_df[~coef_df["is_utility"]].head(10)
    for _, row in non_utility_top.iterrows():
        direction = "raises CT win probability" if row["coefficient"] > 0 else "lowers CT win probability"
        summary_lines.append(
            f"- `{row['feature']}`: coefficient `{row['coefficient']:.6f}` ({direction})"
        )

    contribution_path = None
    if x_round_flat is not None and len(round_df) == len(x_round_flat):
        coef_values = ridge_model.coef_.astype(float)
        contribution_rows = []
        ordered_jump_indices = (
            round_df["delta_from_prev"].abs().sort_values(ascending=False).index.tolist()
        )
        for row_index in ordered_jump_indices:
            position = int(round_df.index.get_loc(row_index))
            if position == 0:
                continue
            feature_delta = x_round_flat[position] - x_round_flat[position - 1]
            contributions = coef_values * feature_delta
            top_contrib_indices = np.argsort(np.abs(contributions))[::-1][:20]
            for rank, feature_index in enumerate(top_contrib_indices, start=1):
                feature_name = flat_feature_names[int(feature_index)]
                base_feature = feature_name.split("__", 1)[1] if "__" in feature_name else feature_name
                contribution_rows.append(
                    {
                        "tick": int(round_df.loc[row_index, "tick"]),
                        "seconds_in_round": round_df.loc[row_index, "seconds_in_round"],
                        "lstm_delta_from_prev": float(round_df.loc[row_index, "delta_from_prev"]),
                        "rank": rank,
                        "feature": feature_name,
                        "base_feature": base_feature,
                        "is_utility": bool(is_utility_column(base_feature)),
                        "feature_delta_scaled": float(feature_delta[int(feature_index)]),
                        "coefficient": float(coef_values[int(feature_index)]),
                        "ridge_delta_contribution": float(contributions[int(feature_index)]),
                    }
                )
        contribution_df = pd.DataFrame(contribution_rows)
        contribution_path = output_dir / "local_round_jump_contributions.csv"
        contribution_df.to_csv(contribution_path, index=False)

        summary_lines.extend(["", "## Largest Jump Contribution Breakdown", ""])
        for row_index in ordered_jump_indices[:5]:
            position = int(round_df.index.get_loc(row_index))
            if position == 0:
                continue
            tick = int(round_df.loc[row_index, "tick"])
            seconds_value = round_df.loc[row_index, "seconds_in_round"]
            seconds_text = "NA" if pd.isna(seconds_value) else f"{float(seconds_value):.2f}"
            lstm_delta = float(round_df.loc[row_index, "delta_from_prev"])
            jump_df = contribution_df[contribution_df["tick"] == tick].copy()
            summary_lines.append("")
            summary_lines.append(
                f"### tick `{tick}`, seconds `{seconds_text}`, LSTM delta `{lstm_delta:+.4f}`"
            )
            summary_lines.append("")
            summary_lines.append("Top all feature movements:")
            for _, contrib_row in jump_df.head(5).iterrows():
                summary_lines.append(
                    f"- `{contrib_row['feature']}`: contribution "
                    f"`{contrib_row['ridge_delta_contribution']:+.6f}`"
                )
            summary_lines.append("")
            summary_lines.append("Top utility-only movements:")
            utility_jump_df = jump_df[jump_df["is_utility"]].head(5)
            if utility_jump_df.empty:
                summary_lines.append("- No utility movement among the top local contributors.")
            else:
                for _, contrib_row in utility_jump_df.iterrows():
                    summary_lines.append(
                        f"- `{contrib_row['feature']}`: contribution "
                        f"`{contrib_row['ridge_delta_contribution']:+.6f}`"
                    )
    summary_path = output_dir / "local_round_explainability_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    timeline_path = output_dir / "local_round_predictions.csv"
    round_df.to_csv(timeline_path, index=False)

    reports = {
        "local_round_ridge_coefficients": str(coef_path),
        "local_round_predictions": str(timeline_path),
        "local_round_explainability_summary": str(summary_path),
    }
    if contribution_path is not None:
        reports["local_round_jump_contributions"] = str(contribution_path)
    return reports


def main() -> None:
    overall_start = time.time()
    args = parse_args()
    validate_split_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
    validate_sampling_args(args.sample_csv_ratio, args.row_stride)
    validate_sampling_args(args.window_sample_ratio, 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_progress("LSTM pipeline indult")
    device = resolve_torch_device(args.device)
    device_info = describe_torch_device(device)
    log_progress(
        "PyTorch eszkoz: "
        f"{device_info['selected']}, cuda_available={device_info['cuda_available']}, "
        f"gpu={device_info['gpu_name']}"
    )

    stage_start = time.time()
    sampled_manifest_rows, split_summaries, sampled_split_summaries = build_or_load_manifest(args)
    pd.DataFrame(sampled_manifest_rows).to_csv(output_dir / "sampled_split_manifest.csv", index=False)
    log_progress(
        f"Manifest kesz {time.time() - stage_start:.1f}s alatt, sorok: {len(sampled_manifest_rows)}"
    )

    data_root = Path(args.data_root)
    split_to_paths = {
        split_name: [
            normalize_csv_path(row["csv_path"], data_root=data_root)
            for row in sampled_manifest_rows
            if row["split"] == split_name
        ]
        for split_name in ["train", "valid", "test"]
    }
    if not split_to_paths["train"] or not split_to_paths["valid"] or not split_to_paths["test"]:
        raise ValueError("Legalabb egy split ures lett.")
    log_progress(
        f"Split CSV darabok: train={len(split_to_paths['train'])}, "
        f"valid={len(split_to_paths['valid'])}, test={len(split_to_paths['test'])}"
    )

    stage_start = time.time()
    feature_names, dropped_features = build_feature_names_from_headers(
        csv_paths=split_to_paths["train"],
        include_tick=args.include_tick,
        drop_utility_features=args.drop_utility_features,
        drop_strong_non_utility_features=args.drop_strong_non_utility_features,
    )
    log_progress(
        f"Feature union kesz {time.time() - stage_start:.1f}s alatt, feature_count={len(feature_names)}"
    )

    stage_start = time.time()
    scaler_mean, scaler_std = compute_scaler_stats(
        csv_paths=split_to_paths["train"],
        feature_names=feature_names,
        row_stride=args.row_stride,
    )
    log_progress(f"Scaler kesz {time.time() - stage_start:.1f}s alatt")

    stage_start = time.time()
    train_infos = build_stream_file_infos(
        split_to_paths["train"], args.row_stride, args.window_sample_ratio, args.random_seed
    )
    valid_infos = build_stream_file_infos(
        split_to_paths["valid"], args.row_stride, args.window_sample_ratio, args.random_seed + 1000
    )
    test_infos = build_stream_file_infos(
        split_to_paths["test"], args.row_stride, args.window_sample_ratio, args.random_seed + 2000
    )
    log_progress(f"Streaming split infok kesz {time.time() - stage_start:.1f}s alatt")

    log_progress("PyTorch LSTM tanitas indul")

    trial_results = []
    best = None
    for trial_idx, trial in enumerate(make_trial_grid(args), start=1):
        trial_start = time.time()
        log_progress(f"Trial {trial_idx} indul: {json.dumps(trial)}")
        model = build_lstm_model(
            (args.sequence_length, len(feature_names)),
            trial,
        )
        train_sequence = StreamingSequence(
            file_infos=train_infos,
            feature_names=feature_names,
            sequence_length=args.sequence_length,
            row_stride=args.row_stride,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            batch_size=trial["batch_size"],
        )
        valid_sequence = StreamingSequence(
            file_infos=valid_infos,
            feature_names=feature_names,
            sequence_length=args.sequence_length,
            row_stride=args.row_stride,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            batch_size=trial["batch_size"],
        )
        model, history = train_torch_lstm_model(
            model=model,
            train_sequence=train_sequence,
            valid_sequence=valid_sequence,
            trial=trial,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
        )
        log_progress(
            f"Trial {trial_idx} tanitas kesz {time.time() - trial_start:.1f}s alatt, "
            f"epochs={len(history.get('loss', []))}"
        )
        train_prob, y_train, train_metadata = predict_split_streaming_torch(
            model,
            train_infos,
            feature_names,
            args.sequence_length,
            args.row_stride,
            scaler_mean,
            scaler_std,
            device,
        )
        valid_prob, y_valid, valid_metadata = predict_split_streaming_torch(
            model,
            valid_infos,
            feature_names,
            args.sequence_length,
            args.row_stride,
            scaler_mean,
            scaler_std,
            device,
        )
        test_prob, y_test, test_metadata = predict_split_streaming_torch(
            model,
            test_infos,
            feature_names,
            args.sequence_length,
            args.row_stride,
            scaler_mean,
            scaler_std,
            device,
        )
        result = {
            "trial_index": trial_idx,
            "params": trial,
            "epochs_ran": len(history.get("loss", [])),
            "best_val_loss": float(np.min(history.get("val_loss", [np.inf]))),
            "train": compute_binary_metrics(y_train, train_prob),
            "valid": compute_binary_metrics(y_valid, valid_prob),
            "test": compute_binary_metrics(y_test, test_prob),
        }
        trial_results.append(result)
        log_progress(
            f"Trial {trial_idx} metrikak: valid_logloss={result['valid']['logloss']:.5f}, "
            f"test_logloss={result['test']['logloss']:.5f}"
        )
        if best is None or result["valid"]["logloss"] < best["result"]["valid"]["logloss"]:
            best = {
                "result": result,
                "model": model,
                "train_prob": train_prob,
                "valid_prob": valid_prob,
                "test_prob": test_prob,
                "train_y": y_train,
                "valid_y": y_valid,
                "test_y": y_test,
                "train_metadata": train_metadata,
                "valid_metadata": valid_metadata,
                "test_metadata": test_metadata,
            }
            log_progress(f"Trial {trial_idx} uj legjobb modell")

    if best is None:
        raise ValueError("Nem sikerult LSTM trialt futtatni.")

    best_model = best["model"]
    import torch

    torch.save(best_model.state_dict(), output_dir / "best_lstm_model.pt")
    log_progress("Legjobb LSTM modell elmentve")

    plot_artifacts = {}
    for split_name in ["train", "valid", "test"]:
        plot_artifacts[split_name] = save_evaluation_plots(
            output_dir=output_dir,
            split_name=f"lstm_{split_name}",
            split_metrics=best["result"][split_name],
        )

    split_payloads = {
        "train": {"y": best["train_y"], "metadata": best["train_metadata"], "prob": best["train_prob"]},
        "valid": {"y": best["valid_y"], "metadata": best["valid_metadata"], "prob": best["valid_prob"]},
        "test": {"y": best["test_y"], "metadata": best["test_metadata"], "prob": best["test_prob"]},
    }

    local_payload = split_payloads[args.local_explain_split]
    local_round_df, local_round_info = pick_local_round(
        metadata_df=local_payload["metadata"],
        probabilities=local_payload["prob"],
        split_name=args.local_explain_split,
        requested_csv=args.local_explain_csv,
        requested_round_num=args.local_explain_round_num,
    )
    log_progress(
        f"Lokalis round valasztva: split={local_round_info['split']}, "
        f"round={local_round_info['round_num']}, csv={Path(local_round_info['csv_path']).name}"
    )
    local_mask = (
        (local_payload["metadata"]["csv_path"] == local_round_info["csv_path"])
        & (local_payload["metadata"]["round_num"] == local_round_info["round_num"])
    ).to_numpy()
    y_local_true = local_payload["y"][local_mask]
    y_local_prob = np.asarray(local_payload["prob"], dtype=np.float64)[local_mask]
    x_local, _, _ = load_local_round_sequences(
        csv_path=Path(local_round_info["csv_path"]),
        round_num=int(local_round_info["round_num"]),
        feature_names=feature_names,
        sequence_length=args.sequence_length,
        row_stride=args.row_stride,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        selected_ticks=local_round_df["tick"].tolist(),
    )
    x_local_flat, flat_feature_names = flatten_sequences(x_local, feature_names)

    local_ridge_model, local_ridge_selection = fit_local_ridge_for_round(
        x_round_flat=x_local_flat,
        y_round_prob=y_local_prob,
        alphas=parse_numeric_list(args.ridge_alphas, float),
    )
    local_ridge_prob = np.clip(local_ridge_model.predict(x_local_flat), 0.0, 1.0)
    local_round_df = local_round_df.copy()
    local_round_df["ridge_probability"] = local_ridge_prob
    local_round_df["delta_from_prev"] = local_round_df["lstm_probability"].diff().fillna(0.0)
    local_round_df["ridge_delta_from_prev"] = local_round_df["ridge_probability"].diff().fillna(0.0)

    local_round_metrics = {
        "fidelity_to_lstm": probability_fidelity(y_local_prob, local_ridge_prob),
        "ridge_vs_true_labels": compute_binary_metrics(y_local_true, local_ridge_prob),
        "lstm_vs_true_labels": compute_binary_metrics(y_local_true, y_local_prob),
        "selection": {
            **local_round_info,
            **local_ridge_selection,
            "row_count": int(len(local_round_df)),
        },
    }
    local_round_reports = write_local_round_reports(
        output_dir=output_dir,
        round_df=local_round_df,
        ridge_model=local_ridge_model,
        flat_feature_names=flat_feature_names,
        x_round_flat=x_local_flat,
    )
    log_progress("Lokalis ridge explainability kesz")

    metrics = {
        "model_type": "lstm_sequence_classifier",
        "device": device_info,
        "sequence_length": args.sequence_length,
        "trial_results": trial_results,
        "best_trial": best["result"],
        "plot_artifacts": plot_artifacts,
        "local_round_explainability": {
            "metrics": local_round_metrics,
            "reports": local_round_reports,
        },
        "feature_count": len(feature_names),
        "features": feature_names,
        "dropped_features": dropped_features,
        "sampling": {
            "manifest_path": args.manifest_path or None,
            "include_partial_csvs": args.include_partial_csvs,
            "sample_csv_ratio": args.sample_csv_ratio,
            "row_stride": args.row_stride,
            "window_sample_ratio": args.window_sample_ratio,
        },
        "splits": split_summaries,
        "sampled_splits": sampled_split_summaries,
        "sequence_counts": {
            "train": int(len(best["train_y"])),
            "valid": int(len(best["valid_y"])),
            "test": int(len(best["test_y"])),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"LSTM modell mentve: {output_dir / 'best_lstm_model.pt'}")
    print(f"Metrikak mentve: {output_dir / 'metrics.json'}")
    print(json.dumps(best['result']['test'], indent=2))
    log_progress(f"Teljes futas kesz {time.time() - overall_start:.1f}s alatt")


if __name__ == "__main__":
    main()
