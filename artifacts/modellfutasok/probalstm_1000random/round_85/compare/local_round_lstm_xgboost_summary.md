# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-b8-vs-wildcard-bo3-EO1cCePneo0X8r6rxB_BMC/b8-vs-wildcard-m2-dust2.csv`
- round_num: `1`
- rows: `145`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.375305 | 0.162836 | 0.498757 | 0.682759 | 0.375305 |
| xgboost | 0.467576 | 0.234967 | 0.663958 | 0.503448 | 0.467576 |

## Closer Per Tick

- lstm: `112`
- xgboost: `33`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
