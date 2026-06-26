# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `2`
- rows: `265`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.136110 | 0.041493 | 0.164482 | 1.000000 | 0.136110 |
| xgboost | 0.194735 | 0.055941 | 0.230849 | 1.000000 | 0.194735 |

## Closer Per Tick

- lstm: `229`
- xgboost: `36`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
