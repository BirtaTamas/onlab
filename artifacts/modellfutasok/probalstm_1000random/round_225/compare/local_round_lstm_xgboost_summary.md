# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-nrg-vs-aurora-bo3-qymu5EnF_DYwHSVf1aSLaG/nrg-vs-aurora-m1-inferno.csv`
- round_num: `14`
- rows: `235`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.100596 | 0.028483 | 0.118692 | 1.000000 | 0.100596 |
| xgboost | 0.156122 | 0.062899 | 0.201051 | 1.000000 | 0.156122 |

## Closer Per Tick

- lstm: `207`
- xgboost: `28`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
