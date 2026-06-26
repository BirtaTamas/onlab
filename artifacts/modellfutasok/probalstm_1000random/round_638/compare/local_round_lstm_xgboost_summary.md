# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m2-mirage.csv`
- round_num: `14`
- rows: `120`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.045604 | 0.002341 | 0.046821 | 1.000000 | 0.954396 |
| xgboost | 0.017928 | 0.000341 | 0.018100 | 1.000000 | 0.982072 |

## Closer Per Tick

- lstm: `0`
- xgboost: `120`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
