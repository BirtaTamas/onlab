# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `16`
- rows: `166`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.270967 | 0.096898 | 0.340771 | 0.903614 | 0.270967 |
| xgboost | 0.343244 | 0.138455 | 0.444688 | 0.765060 | 0.343244 |

## Closer Per Tick

- lstm: `150`
- xgboost: `16`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
