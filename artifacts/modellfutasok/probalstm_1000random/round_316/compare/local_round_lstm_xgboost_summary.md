# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `4`
- rows: `255`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.393619 | 0.179552 | 0.534603 | 0.631373 | 0.393619 |
| xgboost | 0.545510 | 0.311989 | 0.834180 | 0.192157 | 0.545510 |

## Closer Per Tick

- lstm: `250`
- xgboost: `5`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
