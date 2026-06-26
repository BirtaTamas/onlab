# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-heroic-bo3-ReZhZ3UThZvWjRyUeuYiIR/falcons-vs-heroic-m3-dust2.csv`
- round_num: `8`
- rows: `257`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.353393 | 0.205454 | 0.532511 | 0.428016 | 0.353393 |
| xgboost | 0.342716 | 0.185329 | 0.495932 | 0.420233 | 0.342716 |

## Closer Per Tick

- lstm: `123`
- xgboost: `134`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
