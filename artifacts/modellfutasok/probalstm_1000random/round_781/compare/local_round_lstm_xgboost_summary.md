# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `9`
- rows: `124`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.057902 | 0.005975 | 0.061144 | 1.000000 | 0.057902 |
| xgboost | 0.146670 | 0.034422 | 0.167314 | 1.000000 | 0.146670 |

## Closer Per Tick

- lstm: `124`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
