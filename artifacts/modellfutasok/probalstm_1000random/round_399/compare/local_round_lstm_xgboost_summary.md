# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m2-ancient.csv`
- round_num: `5`
- rows: `202`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.108313 | 0.049771 | 0.147964 | 0.851485 | 0.108313 |
| xgboost | 0.107698 | 0.046413 | 0.143798 | 0.856436 | 0.107698 |

## Closer Per Tick

- lstm: `149`
- xgboost: `53`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
