# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `21`
- rows: `250`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.246440 | 0.108194 | 0.325969 | 0.984000 | 0.246440 |
| xgboost | 0.283332 | 0.133068 | 0.385388 | 0.740000 | 0.283332 |

## Closer Per Tick

- lstm: `239`
- xgboost: `11`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
