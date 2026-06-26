# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-3dmax-bo3-NhOpC3bR-AJd86c-60IeuJ/the-mongolz-vs-3dmax-m1-nuke.csv`
- round_num: `8`
- rows: `155`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.343159 | 0.134413 | 0.437641 | 1.000000 | 0.656841 |
| xgboost | 0.320824 | 0.126982 | 0.410764 | 1.000000 | 0.679176 |

## Closer Per Tick

- lstm: `60`
- xgboost: `95`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
