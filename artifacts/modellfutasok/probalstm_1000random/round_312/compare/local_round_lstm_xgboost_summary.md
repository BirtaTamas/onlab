# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `1`
- rows: `122`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.490749 | 0.304330 | 0.794394 | 0.327869 | 0.509251 |
| xgboost | 0.368868 | 0.185341 | 0.522379 | 0.442623 | 0.631132 |

## Closer Per Tick

- lstm: `18`
- xgboost: `104`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
