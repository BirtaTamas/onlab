# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `13`
- rows: `117`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.603895 | 0.461036 | 1.507825 | 0.393162 | 0.396105 |
| xgboost | 0.487463 | 0.323020 | 0.923036 | 0.512821 | 0.512537 |

## Closer Per Tick

- lstm: `1`
- xgboost: `116`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
