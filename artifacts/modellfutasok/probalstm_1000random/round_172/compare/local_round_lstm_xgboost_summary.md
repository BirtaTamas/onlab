# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `15`
- rows: `166`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.008673 | 0.000182 | 0.008766 | 1.000000 | 0.008673 |
| xgboost | 0.024104 | 0.001302 | 0.024786 | 1.000000 | 0.024104 |

## Closer Per Tick

- lstm: `166`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
