# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-vitality-vs-falcons-bo3-8ZTMZQ0BkOa0azICXTbCYv/vitality-vs-falcons-m2-train.csv`
- round_num: `16`
- rows: `142`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.148742 | 0.028901 | 0.165791 | 1.000000 | 0.851258 |
| xgboost | 0.140179 | 0.034249 | 0.161722 | 1.000000 | 0.859821 |

## Closer Per Tick

- lstm: `48`
- xgboost: `94`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
