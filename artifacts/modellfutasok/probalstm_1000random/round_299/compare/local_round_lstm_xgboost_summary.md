# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-the-mongolz-vs-heroic-bo3-lz59_87ZRvJjbdTai7Ev35/heroic-vs-3dmax-m3-ancient.csv`
- round_num: `6`
- rows: `165`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.427301 | 0.219844 | 0.627464 | 0.733333 | 0.572699 |
| xgboost | 0.388636 | 0.186493 | 0.536270 | 0.878788 | 0.611364 |

## Closer Per Tick

- lstm: `31`
- xgboost: `134`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
