# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `8`
- rows: `124`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.301871 | 0.123921 | 0.392515 | 0.870968 | 0.301871 |
| xgboost | 0.331105 | 0.145580 | 0.441703 | 0.677419 | 0.331105 |

## Closer Per Tick

- lstm: `94`
- xgboost: `30`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
