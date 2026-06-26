# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `6`
- rows: `173`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.207467 | 0.064252 | 0.252360 | 0.919075 | 0.792533 |
| xgboost | 0.208911 | 0.063128 | 0.250966 | 1.000000 | 0.791089 |

## Closer Per Tick

- lstm: `80`
- xgboost: `93`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
