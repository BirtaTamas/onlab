# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m3-ancient.csv`
- round_num: `18`
- rows: `154`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.092353 | 0.020566 | 0.105380 | 1.000000 | 0.092353 |
| xgboost | 0.180326 | 0.041548 | 0.205680 | 1.000000 | 0.180326 |

## Closer Per Tick

- lstm: `143`
- xgboost: `11`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
