# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-vitality-vs-mouz-bo3-kZzxcq2ibUgPOmQh0hZOgn/vitality-vs-mouz-m2-train.csv`
- round_num: `9`
- rows: `285`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.445298 | 0.224837 | 0.655932 | 0.750877 | 0.554702 |
| xgboost | 0.352754 | 0.148179 | 0.473503 | 0.863158 | 0.647246 |

## Closer Per Tick

- lstm: `11`
- xgboost: `274`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
