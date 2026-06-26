# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m1-dust2.csv`
- round_num: `5`
- rows: `287`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.487496 | 0.277189 | 0.829742 | 0.620209 | 0.512504 |
| xgboost | 0.511470 | 0.305166 | 0.873047 | 0.337979 | 0.488530 |

## Closer Per Tick

- lstm: `204`
- xgboost: `83`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
