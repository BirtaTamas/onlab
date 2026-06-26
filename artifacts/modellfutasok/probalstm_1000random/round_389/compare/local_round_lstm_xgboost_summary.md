# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `10`
- rows: `205`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.259571 | 0.153508 | 0.409242 | 0.697561 | 0.259571 |
| xgboost | 0.316679 | 0.204848 | 0.558944 | 0.687805 | 0.316679 |

## Closer Per Tick

- lstm: `194`
- xgboost: `11`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
