# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `2`
- rows: `136`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.070793 | 0.005929 | 0.073952 | 1.000000 | 0.929207 |
| xgboost | 0.019927 | 0.000460 | 0.020161 | 1.000000 | 0.980073 |

## Closer Per Tick

- lstm: `0`
- xgboost: `136`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
