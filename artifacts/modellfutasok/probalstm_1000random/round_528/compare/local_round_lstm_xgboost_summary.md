# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `8`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.564536 | 0.340413 | 0.925346 | 0.386957 | 0.435464 |
| xgboost | 0.487249 | 0.254488 | 0.712750 | 0.634783 | 0.512751 |

## Closer Per Tick

- lstm: `9`
- xgboost: `221`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
