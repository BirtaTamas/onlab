# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `3`
- rows: `140`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.003620 | 0.000034 | 0.003638 | 1.000000 | 0.003620 |
| xgboost | 0.011681 | 0.000518 | 0.011950 | 1.000000 | 0.011681 |

## Closer Per Tick

- lstm: `131`
- xgboost: `9`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
