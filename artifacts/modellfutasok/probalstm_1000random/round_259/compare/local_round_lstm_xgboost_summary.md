# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-natus-vincere-vs-3dmax-bo3-JB3JZO-5zNCohi5tAgyHtq/natus-vincere-vs-3dmax-m2-inferno.csv`
- round_num: `4`
- rows: `195`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.073519 | 0.009934 | 0.079042 | 1.000000 | 0.073519 |
| xgboost | 0.109822 | 0.022637 | 0.123215 | 1.000000 | 0.109822 |

## Closer Per Tick

- lstm: `169`
- xgboost: `26`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
