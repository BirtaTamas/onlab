# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-vitality-bo5-g3-5jFl1QSVPqll-eeCKIE/mouz-vs-vitality-m1-inferno.csv`
- round_num: `4`
- rows: `274`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.342198 | 0.186137 | 0.497014 | 0.503650 | 0.342198 |
| xgboost | 0.388765 | 0.224457 | 0.588458 | 0.463504 | 0.388765 |

## Closer Per Tick

- lstm: `263`
- xgboost: `11`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
