# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `3`
- rows: `223`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.463751 | 0.230300 | 0.648678 | 0.587444 | 0.463751 |
| xgboost | 0.475224 | 0.232275 | 0.659682 | 0.825112 | 0.475224 |

## Closer Per Tick

- lstm: `69`
- xgboost: `154`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
