# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `11`
- rows: `184`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.243339 | 0.083907 | 0.302002 | 1.000000 | 0.756661 |
| xgboost | 0.256133 | 0.102445 | 0.332420 | 0.945652 | 0.743867 |

## Closer Per Tick

- lstm: `98`
- xgboost: `86`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
