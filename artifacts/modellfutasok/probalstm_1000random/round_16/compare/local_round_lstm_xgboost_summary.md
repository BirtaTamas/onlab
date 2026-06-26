# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-gamerlegion-vs-tyloo-bo3-CHuj0-KFwAe9c3Zh96vlUq/gamerlegion-vs-tyloo-m2-ancient.csv`
- round_num: `12`
- rows: `177`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.165174 | 0.051516 | 0.200394 | 1.000000 | 0.834826 |
| xgboost | 0.160359 | 0.059361 | 0.203313 | 1.000000 | 0.839641 |

## Closer Per Tick

- lstm: `57`
- xgboost: `120`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
