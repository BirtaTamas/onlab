# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-spirit-vs-astralis-bo3-GZVTrKsE-zdG9dH6juITei/spirit-vs-astralis-m1-nuke.csv`
- round_num: `5`
- rows: `182`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.417420 | 0.224308 | 0.600238 | 0.236264 | 0.417420 |
| xgboost | 0.397520 | 0.202163 | 0.557039 | 0.346154 | 0.397520 |

## Closer Per Tick

- lstm: `61`
- xgboost: `121`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
