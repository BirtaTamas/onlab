# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `2`
- rows: `132`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.125991 | 0.018212 | 0.136178 | 1.000000 | 0.874009 |
| xgboost | 0.016815 | 0.000295 | 0.016964 | 1.000000 | 0.983185 |

## Closer Per Tick

- lstm: `0`
- xgboost: `132`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
