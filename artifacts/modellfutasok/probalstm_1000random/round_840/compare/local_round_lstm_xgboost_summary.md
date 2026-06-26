# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `2`
- rows: `243`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.244918 | 0.068332 | 0.288631 | 0.983539 | 0.244918 |
| xgboost | 0.304970 | 0.102755 | 0.375095 | 0.971193 | 0.304970 |

## Closer Per Tick

- lstm: `198`
- xgboost: `45`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
