# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m3-overpass.csv`
- round_num: `11`
- rows: `217`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.149992 | 0.034027 | 0.171078 | 1.000000 | 0.850008 |
| xgboost | 0.120632 | 0.024514 | 0.135843 | 1.000000 | 0.879368 |

## Closer Per Tick

- lstm: `61`
- xgboost: `156`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
