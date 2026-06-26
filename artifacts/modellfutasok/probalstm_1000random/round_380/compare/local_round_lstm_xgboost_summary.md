# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `6`
- rows: `305`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.366531 | 0.187914 | 0.515658 | 0.511475 | 0.366531 |
| xgboost | 0.358643 | 0.179256 | 0.498194 | 0.557377 | 0.358643 |

## Closer Per Tick

- lstm: `149`
- xgboost: `156`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
