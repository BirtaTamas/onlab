# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `3`
- rows: `181`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.415361 | 0.177003 | 0.542706 | 0.972376 | 0.584639 |
| xgboost | 0.480604 | 0.239567 | 0.667394 | 0.569061 | 0.519396 |

## Closer Per Tick

- lstm: `141`
- xgboost: `40`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
