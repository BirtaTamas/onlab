# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-eternal-fire-vs-spirit-bo5-7H36TpK_LYGHtCXpF3Cgdr/eternal-fire-vs-spirit-m3-dust2.csv`
- round_num: `5`
- rows: `163`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.417587 | 0.193167 | 0.566836 | 0.736196 | 0.582413 |
| xgboost | 0.438775 | 0.213002 | 0.611065 | 0.361963 | 0.561225 |

## Closer Per Tick

- lstm: `106`
- xgboost: `57`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
