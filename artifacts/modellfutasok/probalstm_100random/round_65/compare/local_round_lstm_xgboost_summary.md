# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv`
- round_num: `5`
- rows: `231`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.179379 | 0.063451 | 0.225103 | 0.939394 | 0.179379 |
| xgboost | 0.300962 | 0.107659 | 0.375846 | 0.926407 | 0.300962 |

## Closer Per Tick

- lstm: `197`
- xgboost: `34`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
