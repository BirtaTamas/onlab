# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m3-mirage.csv`
- round_num: `14`
- rows: `290`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.714933 | 0.549486 | 1.558548 | 0.086207 | 0.285067 |
| xgboost | 0.602961 | 0.384081 | 0.980393 | 0.089655 | 0.397039 |

## Closer Per Tick

- lstm: `51`
- xgboost: `239`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
