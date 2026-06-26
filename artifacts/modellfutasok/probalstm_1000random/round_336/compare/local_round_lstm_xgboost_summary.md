# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-872ZDvS9tk2PrtGeXVe8dJ/aurora-vs-heroic-m1-train-p3.csv`
- round_num: `3`
- rows: `144`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.061330 | 0.007772 | 0.065796 | 1.000000 | 0.061330 |
| xgboost | 0.157516 | 0.039217 | 0.182349 | 1.000000 | 0.157516 |

## Closer Per Tick

- lstm: `144`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
