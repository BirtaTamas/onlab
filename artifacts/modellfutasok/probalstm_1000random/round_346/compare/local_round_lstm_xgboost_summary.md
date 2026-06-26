# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `23`
- rows: `241`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.391796 | 0.190893 | 0.541928 | 0.755187 | 0.391796 |
| xgboost | 0.455110 | 0.252732 | 0.680328 | 0.522822 | 0.455110 |

## Closer Per Tick

- lstm: `226`
- xgboost: `15`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
