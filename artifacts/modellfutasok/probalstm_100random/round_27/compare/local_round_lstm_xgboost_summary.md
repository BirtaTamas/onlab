# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `7`
- rows: `167`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.176891 | 0.073501 | 0.231122 | 0.922156 | 0.176891 |
| xgboost | 0.249182 | 0.113383 | 0.337241 | 0.874251 | 0.249182 |

## Closer Per Tick

- lstm: `161`
- xgboost: `6`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
