# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-mouz-vs-falcons-bo3-ET1FlQ7LAGQtcSrRzzPcv6/mouz-vs-falcons-m1-dust2.csv`
- round_num: `12`
- rows: `125`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.242289 | 0.070301 | 0.287184 | 1.000000 | 0.757711 |
| xgboost | 0.245918 | 0.072323 | 0.292019 | 1.000000 | 0.754082 |

## Closer Per Tick

- lstm: `62`
- xgboost: `63`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
