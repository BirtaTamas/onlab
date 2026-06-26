# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-virtuspro-vs-spirit-bo3-KJqZR5yNeHXaNsc7MGaDWB/virtus-pro-vs-spirit-m1-train.csv`
- round_num: `7`
- rows: `133`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.258285 | 0.069151 | 0.300811 | 1.000000 | 0.741715 |
| xgboost | 0.211199 | 0.046634 | 0.238737 | 1.000000 | 0.788801 |

## Closer Per Tick

- lstm: `4`
- xgboost: `129`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
