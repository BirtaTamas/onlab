# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-vitality-bo5-g3-5jFl1QSVPqll-eeCKIE/mouz-vs-vitality-m1-inferno.csv`
- round_num: `12`
- rows: `177`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.174054 | 0.063144 | 0.218030 | 1.000000 | 0.174054 |
| xgboost | 0.240163 | 0.102297 | 0.318250 | 0.711864 | 0.240163 |

## Closer Per Tick

- lstm: `177`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
