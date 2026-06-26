# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `10`
- rows: `225`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.366308 | 0.149058 | 0.476330 | 0.782222 | 0.633692 |
| xgboost | 0.342083 | 0.129587 | 0.432969 | 0.955556 | 0.657917 |

## Closer Per Tick

- lstm: `70`
- xgboost: `155`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
