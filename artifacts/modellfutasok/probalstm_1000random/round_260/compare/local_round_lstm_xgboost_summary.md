# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `10`
- rows: `245`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.580072 | 0.437695 | 1.086015 | 0.236735 | 0.580072 |
| xgboost | 0.546664 | 0.386359 | 0.953394 | 0.248980 | 0.546664 |

## Closer Per Tick

- lstm: `61`
- xgboost: `184`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
