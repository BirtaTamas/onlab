# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m2-ancient.csv`
- round_num: `16`
- rows: `173`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.370580 | 0.161364 | 0.499930 | 0.947977 | 0.629420 |
| xgboost | 0.374962 | 0.161549 | 0.494902 | 0.930636 | 0.625038 |

## Closer Per Tick

- lstm: `97`
- xgboost: `76`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
