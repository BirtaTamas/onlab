# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `25`
- rows: `232`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.499603 | 0.279820 | 0.791844 | 0.750000 | 0.500397 |
| xgboost | 0.474768 | 0.246902 | 0.688482 | 0.775862 | 0.525232 |

## Closer Per Tick

- lstm: `100`
- xgboost: `132`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
