# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-astralis-bo3-Zley6FZuKcttfrliAqsvWJ/astralis-vs-vitality-m1-inferno.csv`
- round_num: `13`
- rows: `113`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.249848 | 0.109241 | 0.333027 | 0.716814 | 0.249848 |
| xgboost | 0.324418 | 0.131906 | 0.422652 | 0.734513 | 0.324418 |

## Closer Per Tick

- lstm: `81`
- xgboost: `32`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
