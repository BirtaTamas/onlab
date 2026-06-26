# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `14`
- rows: `288`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.150659 | 0.037119 | 0.173475 | 1.000000 | 0.150659 |
| xgboost | 0.239327 | 0.084433 | 0.295829 | 1.000000 | 0.239327 |

## Closer Per Tick

- lstm: `260`
- xgboost: `28`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
