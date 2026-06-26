# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv`
- round_num: `11`
- rows: `214`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.141621 | 0.044715 | 0.170855 | 1.000000 | 0.141621 |
| xgboost | 0.258031 | 0.106767 | 0.337536 | 1.000000 | 0.258031 |

## Closer Per Tick

- lstm: `214`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
