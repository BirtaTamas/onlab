# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m1-inferno.csv`
- round_num: `13`
- rows: `133`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.211784 | 0.086355 | 0.276880 | 0.766917 | 0.211784 |
| xgboost | 0.206268 | 0.077564 | 0.262888 | 0.909774 | 0.206268 |

## Closer Per Tick

- lstm: `62`
- xgboost: `71`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
