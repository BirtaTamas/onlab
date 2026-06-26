# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `14`
- rows: `215`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.130312 | 0.035532 | 0.152859 | 1.000000 | 0.130312 |
| xgboost | 0.132059 | 0.037818 | 0.157042 | 1.000000 | 0.132059 |

## Closer Per Tick

- lstm: `152`
- xgboost: `63`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
