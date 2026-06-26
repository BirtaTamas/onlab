# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `18`
- rows: `157`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.224233 | 0.081111 | 0.281228 | 1.000000 | 0.775767 |
| xgboost | 0.228824 | 0.092364 | 0.296725 | 1.000000 | 0.771176 |

## Closer Per Tick

- lstm: `59`
- xgboost: `98`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
