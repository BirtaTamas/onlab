# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-legacy-bo3-NvI4DRplwm0O-zy6YVkFbj/wildcard-vs-legacy-m2-nuke.csv`
- round_num: `19`
- rows: `104`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.498126 | 0.270828 | 0.725648 | 0.336538 | 0.498126 |
| xgboost | 0.442009 | 0.221531 | 0.620065 | 0.471154 | 0.442009 |

## Closer Per Tick

- lstm: `13`
- xgboost: `91`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
