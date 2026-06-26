# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `31`
- rows: `179`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.244069 | 0.083173 | 0.301169 | 0.977654 | 0.755931 |
| xgboost | 0.186272 | 0.055399 | 0.223674 | 0.921788 | 0.813728 |

## Closer Per Tick

- lstm: `13`
- xgboost: `166`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
