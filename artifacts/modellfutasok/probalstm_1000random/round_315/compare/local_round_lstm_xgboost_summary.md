# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `8`
- rows: `184`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.043429 | 0.006208 | 0.046915 | 1.000000 | 0.043429 |
| xgboost | 0.046854 | 0.006264 | 0.050324 | 1.000000 | 0.046854 |

## Closer Per Tick

- lstm: `146`
- xgboost: `38`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
