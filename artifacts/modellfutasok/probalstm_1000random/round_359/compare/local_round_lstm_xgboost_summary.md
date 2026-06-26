# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv`
- round_num: `1`
- rows: `115`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.476916 | 0.241457 | 0.669883 | 0.434783 | 0.523084 |
| xgboost | 0.382306 | 0.158402 | 0.496643 | 1.000000 | 0.617694 |

## Closer Per Tick

- lstm: `12`
- xgboost: `103`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
