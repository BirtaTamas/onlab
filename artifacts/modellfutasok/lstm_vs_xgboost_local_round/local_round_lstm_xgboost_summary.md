# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full\iem_chengdu\iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR\heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `21`
- rows: `139`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.429074 | 0.270104 | 0.711699 | 0.388489 | 0.429074 |
| xgboost | 0.493415 | 0.329994 | 0.879126 | 0.338129 | 0.493415 |

## Closer Per Tick

- lstm: `133`
- xgboost: `6`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
