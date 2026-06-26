# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-tyloo-vs-vitality-bo3-aF98ikh3PjdqKlkdIJn9tC/tyloo-vs-vitality-m1-inferno.csv`
- round_num: `8`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.276759 | 0.103227 | 0.347456 | 1.000000 | 0.723241 |
| xgboost | 0.317003 | 0.143512 | 0.424343 | 1.000000 | 0.682997 |

## Closer Per Tick

- lstm: `139`
- xgboost: `91`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
