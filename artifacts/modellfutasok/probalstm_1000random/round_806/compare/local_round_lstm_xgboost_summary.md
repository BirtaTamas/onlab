# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `5`
- rows: `238`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.214067 | 0.090269 | 0.281469 | 0.894958 | 0.214067 |
| xgboost | 0.214926 | 0.084064 | 0.274796 | 0.894958 | 0.214926 |

## Closer Per Tick

- lstm: `162`
- xgboost: `76`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
