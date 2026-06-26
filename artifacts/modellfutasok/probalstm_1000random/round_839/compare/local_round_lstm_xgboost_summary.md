# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-heroic-bo3-ReZhZ3UThZvWjRyUeuYiIR/falcons-vs-heroic-m3-dust2.csv`
- round_num: `15`
- rows: `149`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.312025 | 0.115399 | 0.391727 | 1.000000 | 0.687975 |
| xgboost | 0.274345 | 0.099137 | 0.342353 | 1.000000 | 0.725655 |

## Closer Per Tick

- lstm: `28`
- xgboost: `121`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
