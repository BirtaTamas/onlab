# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv`
- round_num: `5`
- rows: `223`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.470874 | 0.253261 | 0.744820 | 0.834081 | 0.470874 |
| xgboost | 0.541033 | 0.316442 | 0.914774 | 0.327354 | 0.541033 |

## Closer Per Tick

- lstm: `217`
- xgboost: `6`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
