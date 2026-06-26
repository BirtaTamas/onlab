# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-vitality-vs-mouz-bo3-kZzxcq2ibUgPOmQh0hZOgn/vitality-vs-mouz-m2-train.csv`
- round_num: `18`
- rows: `142`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.394302 | 0.203319 | 0.555772 | 0.464789 | 0.394302 |
| xgboost | 0.458343 | 0.271741 | 0.699604 | 0.253521 | 0.458343 |

## Closer Per Tick

- lstm: `140`
- xgboost: `2`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
