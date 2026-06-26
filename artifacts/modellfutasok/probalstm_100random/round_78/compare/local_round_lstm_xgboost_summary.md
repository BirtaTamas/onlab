# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-natus-vincere-vs-3dmax-bo3-JB3JZO-5zNCohi5tAgyHtq/natus-vincere-vs-3dmax-m2-inferno.csv`
- round_num: `7`
- rows: `167`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.476756 | 0.266950 | 0.767573 | 0.622754 | 0.476756 |
| xgboost | 0.557968 | 0.351684 | 0.986943 | 0.526946 | 0.557968 |

## Closer Per Tick

- lstm: `160`
- xgboost: `7`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
