# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `4`
- rows: `173`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.050607 | 0.002761 | 0.052044 | 1.000000 | 0.949393 |
| xgboost | 0.021221 | 0.000467 | 0.021458 | 1.000000 | 0.978779 |

## Closer Per Tick

- lstm: `0`
- xgboost: `173`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
