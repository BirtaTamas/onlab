# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-spirit-vs-the-mongolz-bo3-Ep_2Z5_t0VWYbCORdH0Tlg/spirit-vs-the-mongolz-m3-mirage.csv`
- round_num: `16`
- rows: `157`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.451458 | 0.233780 | 0.681067 | 0.515924 | 0.548542 |
| xgboost | 0.391711 | 0.190908 | 0.550618 | 0.585987 | 0.608289 |

## Closer Per Tick

- lstm: `33`
- xgboost: `124`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
