# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m1-inferno.csv`
- round_num: `10`
- rows: `176`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.169828 | 0.079365 | 0.233930 | 0.818182 | 0.169828 |
| xgboost | 0.157859 | 0.067625 | 0.209344 | 0.761364 | 0.157859 |

## Closer Per Tick

- lstm: `82`
- xgboost: `94`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
