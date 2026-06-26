# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv`
- round_num: `17`
- rows: `161`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.181234 | 0.083019 | 0.247234 | 0.863354 | 0.181234 |
| xgboost | 0.186116 | 0.068504 | 0.235909 | 0.975155 | 0.186116 |

## Closer Per Tick

- lstm: `105`
- xgboost: `56`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
