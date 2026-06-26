# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `13`
- rows: `132`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.580839 | 0.379221 | 1.009543 | 0.545455 | 0.419161 |
| xgboost | 0.428681 | 0.202402 | 0.584276 | 0.765152 | 0.571319 |

## Closer Per Tick

- lstm: `30`
- xgboost: `102`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
