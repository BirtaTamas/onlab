# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `34`
- rows: `265`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.464003 | 0.258170 | 0.687637 | 0.301887 | 0.464003 |
| xgboost | 0.418144 | 0.223355 | 0.605238 | 0.441509 | 0.418144 |

## Closer Per Tick

- lstm: `49`
- xgboost: `216`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
