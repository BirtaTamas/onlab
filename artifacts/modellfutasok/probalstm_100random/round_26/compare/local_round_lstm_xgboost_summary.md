# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-b8-vs-wildcard-bo3-EO1cCePneo0X8r6rxB_BMC/b8-vs-wildcard-m3-inferno.csv`
- round_num: `2`
- rows: `196`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.266435 | 0.081943 | 0.321051 | 0.969388 | 0.733565 |
| xgboost | 0.249371 | 0.072212 | 0.296327 | 0.994898 | 0.750629 |

## Closer Per Tick

- lstm: `49`
- xgboost: `147`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
