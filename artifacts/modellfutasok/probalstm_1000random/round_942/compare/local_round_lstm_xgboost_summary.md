# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m2-mirage.csv`
- round_num: `2`
- rows: `200`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.488559 | 0.256131 | 0.710552 | 0.550000 | 0.511441 |
| xgboost | 0.394062 | 0.167352 | 0.515822 | 0.965000 | 0.605938 |

## Closer Per Tick

- lstm: `2`
- xgboost: `198`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
