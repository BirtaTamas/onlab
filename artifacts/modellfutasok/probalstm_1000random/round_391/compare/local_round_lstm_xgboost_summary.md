# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-virtuspro-vs-spirit-bo3-KJqZR5yNeHXaNsc7MGaDWB/virtus-pro-vs-spirit-m1-train.csv`
- round_num: `13`
- rows: `173`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.460993 | 0.230031 | 0.640902 | 0.294798 | 0.460993 |
| xgboost | 0.511165 | 0.284823 | 0.753673 | 0.213873 | 0.511165 |

## Closer Per Tick

- lstm: `162`
- xgboost: `11`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
