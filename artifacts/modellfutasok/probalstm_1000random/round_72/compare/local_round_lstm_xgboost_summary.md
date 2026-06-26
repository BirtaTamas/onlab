# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-vitality-vs-faze-bo3-hDX5yjYYbla4cw8aPwAYi3/vitality-vs-faze-m1-nuke.csv`
- round_num: `2`
- rows: `206`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.201889 | 0.065338 | 0.245611 | 1.000000 | 0.201889 |
| xgboost | 0.204241 | 0.055740 | 0.239163 | 1.000000 | 0.204241 |

## Closer Per Tick

- lstm: `104`
- xgboost: `102`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
