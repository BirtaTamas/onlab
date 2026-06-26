# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-the-mongolz-vs-natus-vincere-bo3-C0GZxMhpGHBr28LeyjgICZ/the-mongolz-vs-natus-vincere-m1-mirage.csv`
- round_num: `9`
- rows: `221`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.147797 | 0.046440 | 0.178794 | 0.972851 | 0.147797 |
| xgboost | 0.233127 | 0.083928 | 0.291870 | 0.954751 | 0.233127 |

## Closer Per Tick

- lstm: `218`
- xgboost: `3`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
