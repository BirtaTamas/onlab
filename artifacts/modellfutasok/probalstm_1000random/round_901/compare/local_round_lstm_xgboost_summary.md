# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv`
- round_num: `9`
- rows: `223`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.264061 | 0.116454 | 0.360165 | 0.771300 | 0.264061 |
| xgboost | 0.250215 | 0.099494 | 0.328163 | 0.798206 | 0.250215 |

## Closer Per Tick

- lstm: `99`
- xgboost: `124`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
