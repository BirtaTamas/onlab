# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-3dmax-bo3-NhOpC3bR-AJd86c-60IeuJ/the-mongolz-vs-3dmax-m1-nuke.csv`
- round_num: `13`
- rows: `169`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.493167 | 0.279621 | 0.745974 | 0.230769 | 0.493167 |
| xgboost | 0.610865 | 0.408953 | 1.085663 | 0.159763 | 0.610865 |

## Closer Per Tick

- lstm: `168`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
