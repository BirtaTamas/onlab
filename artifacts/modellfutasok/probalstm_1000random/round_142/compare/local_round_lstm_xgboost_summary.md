# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m3-overpass.csv`
- round_num: `3`
- rows: `156`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.011600 | 0.000280 | 0.011743 | 1.000000 | 0.011600 |
| xgboost | 0.040637 | 0.002424 | 0.041906 | 1.000000 | 0.040637 |

## Closer Per Tick

- lstm: `156`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
