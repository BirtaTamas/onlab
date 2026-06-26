# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `31`
- rows: `249`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.370229 | 0.174730 | 0.512557 | 0.843373 | 0.629771 |
| xgboost | 0.396503 | 0.203223 | 0.573909 | 0.566265 | 0.603497 |

## Closer Per Tick

- lstm: `161`
- xgboost: `88`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
