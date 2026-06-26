# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-mouz-vs-falcons-bo3-ET1FlQ7LAGQtcSrRzzPcv6/mouz-vs-falcons-m1-dust2.csv`
- round_num: `7`
- rows: `227`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.164214 | 0.052390 | 0.200539 | 0.982379 | 0.164214 |
| xgboost | 0.218595 | 0.076563 | 0.272838 | 1.000000 | 0.218595 |

## Closer Per Tick

- lstm: `211`
- xgboost: `16`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
