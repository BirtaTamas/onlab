# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-nrg-vs-aurora-bo3-qymu5EnF_DYwHSVf1aSLaG/nrg-vs-aurora-m1-inferno.csv`
- round_num: `15`
- rows: `213`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.127952 | 0.038365 | 0.154144 | 0.985915 | 0.127952 |
| xgboost | 0.100646 | 0.017286 | 0.110586 | 1.000000 | 0.100646 |

## Closer Per Tick

- lstm: `145`
- xgboost: `68`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
