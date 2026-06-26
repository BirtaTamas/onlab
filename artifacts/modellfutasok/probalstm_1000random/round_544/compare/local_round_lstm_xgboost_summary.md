# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `20`
- rows: `304`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.400067 | 0.236990 | 0.611318 | 0.322368 | 0.400067 |
| xgboost | 0.369659 | 0.202457 | 0.540777 | 0.542763 | 0.369659 |

## Closer Per Tick

- lstm: `102`
- xgboost: `202`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
