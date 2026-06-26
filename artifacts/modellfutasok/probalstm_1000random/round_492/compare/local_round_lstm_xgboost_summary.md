# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `17`
- rows: `171`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.425358 | 0.198861 | 0.578717 | 0.479532 | 0.425358 |
| xgboost | 0.449710 | 0.218213 | 0.620353 | 0.438596 | 0.449710 |

## Closer Per Tick

- lstm: `129`
- xgboost: `42`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
