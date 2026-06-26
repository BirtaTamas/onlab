# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `15`
- rows: `143`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.136238 | 0.033730 | 0.158139 | 1.000000 | 0.136238 |
| xgboost | 0.210296 | 0.054563 | 0.244787 | 1.000000 | 0.210296 |

## Closer Per Tick

- lstm: `131`
- xgboost: `12`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
