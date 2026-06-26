# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-gamerlegion-vs-tyloo-bo3-CHuj0-KFwAe9c3Zh96vlUq/gamerlegion-vs-tyloo-m2-ancient.csv`
- round_num: `1`
- rows: `163`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.318491 | 0.145879 | 0.433767 | 0.822086 | 0.681509 |
| xgboost | 0.278714 | 0.132153 | 0.383760 | 0.957055 | 0.721286 |

## Closer Per Tick

- lstm: `13`
- xgboost: `150`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
