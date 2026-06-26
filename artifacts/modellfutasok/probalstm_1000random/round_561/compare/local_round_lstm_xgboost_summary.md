# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-spirit-vs-astralis-bo3-GZVTrKsE-zdG9dH6juITei/spirit-vs-astralis-m1-nuke.csv`
- round_num: `3`
- rows: `296`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.047016 | 0.004141 | 0.049268 | 1.000000 | 0.047016 |
| xgboost | 0.137118 | 0.027182 | 0.153015 | 1.000000 | 0.137118 |

## Closer Per Tick

- lstm: `276`
- xgboost: `20`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
