# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv`
- round_num: `11`
- rows: `210`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.415432 | 0.218622 | 0.638654 | 0.757143 | 0.584568 |
| xgboost | 0.457937 | 0.247080 | 0.686384 | 0.690476 | 0.542063 |

## Closer Per Tick

- lstm: `159`
- xgboost: `51`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
