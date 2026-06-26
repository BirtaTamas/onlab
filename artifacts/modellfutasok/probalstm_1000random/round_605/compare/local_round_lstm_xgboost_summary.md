# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m3-ancient.csv`
- round_num: `8`
- rows: `113`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.223167 | 0.072746 | 0.272499 | 1.000000 | 0.776833 |
| xgboost | 0.243244 | 0.087097 | 0.303355 | 0.973451 | 0.756756 |

## Closer Per Tick

- lstm: `72`
- xgboost: `41`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
