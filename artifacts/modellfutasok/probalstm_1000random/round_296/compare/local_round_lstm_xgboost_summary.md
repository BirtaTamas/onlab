# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-flyquest-bo3-ElcEZT56lTCLJYDcWlMY2d/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `7`
- rows: `203`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.285330 | 0.127383 | 0.380208 | 0.886700 | 0.285330 |
| xgboost | 0.333658 | 0.154829 | 0.452350 | 0.911330 | 0.333658 |

## Closer Per Tick

- lstm: `173`
- xgboost: `30`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
