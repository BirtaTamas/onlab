# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-b8-vs-wildcard-bo3-EO1cCePneo0X8r6rxB_BMC/b8-vs-wildcard-m3-inferno.csv`
- round_num: `4`
- rows: `218`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.127914 | 0.041820 | 0.157522 | 1.000000 | 0.872086 |
| xgboost | 0.119968 | 0.050635 | 0.158300 | 0.917431 | 0.880032 |

## Closer Per Tick

- lstm: `49`
- xgboost: `169`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
