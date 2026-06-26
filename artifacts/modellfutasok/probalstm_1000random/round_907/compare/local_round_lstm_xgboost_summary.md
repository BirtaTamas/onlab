# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `18`
- rows: `264`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.266754 | 0.112992 | 0.350580 | 0.799242 | 0.266754 |
| xgboost | 0.281386 | 0.109679 | 0.359317 | 0.958333 | 0.281386 |

## Closer Per Tick

- lstm: `185`
- xgboost: `79`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
