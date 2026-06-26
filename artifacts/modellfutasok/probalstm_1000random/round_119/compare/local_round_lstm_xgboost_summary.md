# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `15`
- rows: `154`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.178219 | 0.044828 | 0.206661 | 1.000000 | 0.821781 |
| xgboost | 0.188763 | 0.054698 | 0.225228 | 1.000000 | 0.811237 |

## Closer Per Tick

- lstm: `67`
- xgboost: `87`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
