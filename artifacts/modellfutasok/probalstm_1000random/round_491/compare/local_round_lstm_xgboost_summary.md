# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m2-mirage.csv`
- round_num: `5`
- rows: `192`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.199685 | 0.047660 | 0.228789 | 1.000000 | 0.800315 |
| xgboost | 0.173164 | 0.037523 | 0.195652 | 1.000000 | 0.826836 |

## Closer Per Tick

- lstm: `48`
- xgboost: `144`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
