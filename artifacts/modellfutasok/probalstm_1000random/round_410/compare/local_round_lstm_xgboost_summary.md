# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `14`
- rows: `117`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.162049 | 0.040192 | 0.187740 | 1.000000 | 0.162049 |
| xgboost | 0.226405 | 0.061774 | 0.265644 | 1.000000 | 0.226405 |

## Closer Per Tick

- lstm: `109`
- xgboost: `8`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
