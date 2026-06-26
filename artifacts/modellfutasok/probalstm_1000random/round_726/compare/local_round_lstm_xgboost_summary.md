# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-the-mongolz-vs-heroic-bo3-lz59_87ZRvJjbdTai7Ev35/heroic-vs-3dmax-m3-ancient.csv`
- round_num: `8`
- rows: `133`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.141016 | 0.054955 | 0.180785 | 0.984962 | 0.141016 |
| xgboost | 0.238018 | 0.096077 | 0.308264 | 0.864662 | 0.238018 |

## Closer Per Tick

- lstm: `132`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
