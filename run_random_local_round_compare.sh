#!/bin/zsh
set -euo pipefail

LSTM_RUN_DIR="artifacts/modellfutasok/lstm_sequence_full_cuda_final"
XGBOOST_RUN_DIR="artifacts/modellfutasok/xgboost_streaming_final_with_utility_100pct"
DATA_ROOT="processed_full"

RANDOM_CSV="processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-liquid-bo3-5QhocJqNMgLFMdnITD1OfU/mouz-vs-liquid-m3-mirage.csv"
RANDOM_ROUND_NUM="32"

LSTM_LOCAL_OUT="artifacts/modellfutasok/lstm_local_round_random_mouz_liquid_mirage_r32"
COMPARE_OUT="artifacts/modellfutasok/lstm_vs_xgboost_random_mouz_liquid_mirage_r32"

python predict_lstm_local_round.py \
  --lstm-run-dir "$LSTM_RUN_DIR" \
  --data-root "$DATA_ROOT" \
  --csv "$RANDOM_CSV" \
  --round-num "$RANDOM_ROUND_NUM" \
  --output-dir "$LSTM_LOCAL_OUT" \
  --device cuda

python compare_lstm_xgboost_round.py \
  --lstm-run-dir "$LSTM_LOCAL_OUT" \
  --xgboost-run-dir "$XGBOOST_RUN_DIR" \
  --output-dir "$COMPARE_OUT"

python plot_local_round_probabilities.py \
  --input-csv "$COMPARE_OUT/local_round_lstm_xgboost_predictions.csv" \
  --output-png "$COMPARE_OUT/local_round_probability_plot.png" \
  --show-ridge

echo "Random csv: $RANDOM_CSV"
echo "Random round: $RANDOM_ROUND_NUM"
echo "LSTM local output: $LSTM_LOCAL_OUT"
echo "Compare output: $COMPARE_OUT"
