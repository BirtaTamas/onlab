# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `10`
- rows: `199`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.320805 | 0.113786 | 0.398207 | 1.000000 | 0.679195 |
| xgboost | 0.314492 | 0.114126 | 0.393206 | 1.000000 | 0.685508 |

## Closer Per Tick

- lstm: `96`
- xgboost: `103`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
