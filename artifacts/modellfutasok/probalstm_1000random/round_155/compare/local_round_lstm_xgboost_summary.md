# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `17`
- rows: `281`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.135059 | 0.053098 | 0.174267 | 0.911032 | 0.135059 |
| xgboost | 0.131463 | 0.048605 | 0.167737 | 0.846975 | 0.131463 |

## Closer Per Tick

- lstm: `205`
- xgboost: `76`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
