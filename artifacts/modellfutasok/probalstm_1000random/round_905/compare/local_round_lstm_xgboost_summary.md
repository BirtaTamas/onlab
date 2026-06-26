# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `19`
- rows: `123`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.261016 | 0.100785 | 0.330871 | 1.000000 | 0.261016 |
| xgboost | 0.317763 | 0.141436 | 0.421815 | 0.910569 | 0.317763 |

## Closer Per Tick

- lstm: `116`
- xgboost: `7`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
