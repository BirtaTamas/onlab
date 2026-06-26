# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-3dmax-vs-rare-atom-bo3-DWQZo2y3LVjgpuOkyCDf4V/3dmax-vs-rare-atom-m2-ancient.csv`
- round_num: `4`
- rows: `155`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.396283 | 0.182887 | 0.533884 | 0.832258 | 0.603717 |
| xgboost | 0.421737 | 0.211771 | 0.588820 | 0.406452 | 0.578263 |

## Closer Per Tick

- lstm: `116`
- xgboost: `39`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
