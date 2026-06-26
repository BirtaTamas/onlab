# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m2-mirage.csv`
- round_num: `11`
- rows: `139`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.702618 | 0.570726 | 1.512822 | 0.179856 | 0.297382 |
| xgboost | 0.606824 | 0.430411 | 1.066723 | 0.179856 | 0.393176 |

## Closer Per Tick

- lstm: `3`
- xgboost: `136`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
