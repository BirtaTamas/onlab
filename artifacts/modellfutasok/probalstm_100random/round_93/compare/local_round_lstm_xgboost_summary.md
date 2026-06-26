# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `3`
- rows: `177`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.174615 | 0.043963 | 0.201820 | 1.000000 | 0.174615 |
| xgboost | 0.271490 | 0.092672 | 0.334084 | 1.000000 | 0.271490 |

## Closer Per Tick

- lstm: `170`
- xgboost: `7`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
