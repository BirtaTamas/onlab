# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `27`
- rows: `125`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.319743 | 0.185160 | 0.487121 | 0.536000 | 0.319743 |
| xgboost | 0.334587 | 0.207633 | 0.550050 | 0.536000 | 0.334587 |

## Closer Per Tick

- lstm: `89`
- xgboost: `36`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
