# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `21`
- rows: `161`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.132801 | 0.060478 | 0.181611 | 0.807453 | 0.132801 |
| xgboost | 0.151736 | 0.065179 | 0.205304 | 0.807453 | 0.151736 |

## Closer Per Tick

- lstm: `142`
- xgboost: `19`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
