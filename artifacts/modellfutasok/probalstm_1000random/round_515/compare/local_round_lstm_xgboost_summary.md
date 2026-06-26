# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `1`
- rows: `139`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.334313 | 0.138337 | 0.436666 | 0.697842 | 0.334313 |
| xgboost | 0.408007 | 0.194777 | 0.563425 | 0.618705 | 0.408007 |

## Closer Per Tick

- lstm: `136`
- xgboost: `3`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
