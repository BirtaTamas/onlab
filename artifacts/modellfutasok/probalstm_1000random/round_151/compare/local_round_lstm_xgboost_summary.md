# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-nrg-bo3-WMQcRUwgyUmu57EEkX9f3P/falcons-vs-nrg-m1-train.csv`
- round_num: `4`
- rows: `215`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.600320 | 0.385858 | 1.101985 | 0.344186 | 0.399680 |
| xgboost | 0.475707 | 0.240138 | 0.684971 | 0.809302 | 0.524293 |

## Closer Per Tick

- lstm: `1`
- xgboost: `214`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
