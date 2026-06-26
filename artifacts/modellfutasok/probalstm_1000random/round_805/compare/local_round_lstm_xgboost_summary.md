# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv`
- round_num: `2`
- rows: `223`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.146985 | 0.044582 | 0.175653 | 1.000000 | 0.146985 |
| xgboost | 0.161216 | 0.047058 | 0.191166 | 0.995516 | 0.161216 |

## Closer Per Tick

- lstm: `172`
- xgboost: `51`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
