# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m2-mirage.csv`
- round_num: `4`
- rows: `212`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.361435 | 0.138672 | 0.458744 | 0.943396 | 0.638565 |
| xgboost | 0.301348 | 0.110178 | 0.380094 | 0.797170 | 0.698652 |

## Closer Per Tick

- lstm: `47`
- xgboost: `165`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
