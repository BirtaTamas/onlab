# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m1-inferno.csv`
- round_num: `18`
- rows: `115`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.328579 | 0.122300 | 0.412345 | 1.000000 | 0.671421 |
| xgboost | 0.324111 | 0.121287 | 0.407518 | 1.000000 | 0.675889 |

## Closer Per Tick

- lstm: `61`
- xgboost: `54`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
