# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-gamerlegion-vs-liquid-bo3-73g5XINyWmLhIm1c4ZyOM7/gamerlegion-vs-liquid-m1-dust2.csv`
- round_num: `12`
- rows: `194`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.333549 | 0.135542 | 0.431863 | 0.958763 | 0.666451 |
| xgboost | 0.304016 | 0.124803 | 0.396329 | 0.881443 | 0.695984 |

## Closer Per Tick

- lstm: `91`
- xgboost: `103`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
