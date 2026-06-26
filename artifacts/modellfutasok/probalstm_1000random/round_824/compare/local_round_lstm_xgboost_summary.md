# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-natus-vincere-vs-3dmax-bo3-JB3JZO-5zNCohi5tAgyHtq/natus-vincere-vs-3dmax-m2-inferno.csv`
- round_num: `20`
- rows: `228`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.270097 | 0.126898 | 0.365986 | 0.789474 | 0.270097 |
| xgboost | 0.289964 | 0.143527 | 0.401452 | 0.627193 | 0.289964 |

## Closer Per Tick

- lstm: `207`
- xgboost: `21`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
