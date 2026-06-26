# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m2-nuke.csv`
- round_num: `10`
- rows: `131`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.261461 | 0.107756 | 0.342716 | 0.816794 | 0.738539 |
| xgboost | 0.212257 | 0.082535 | 0.271384 | 0.992366 | 0.787743 |

## Closer Per Tick

- lstm: `1`
- xgboost: `130`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
