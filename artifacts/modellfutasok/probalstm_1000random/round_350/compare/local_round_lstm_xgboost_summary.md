# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `7`
- rows: `244`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.165286 | 0.069017 | 0.216110 | 0.938525 | 0.165286 |
| xgboost | 0.191689 | 0.079168 | 0.249996 | 1.000000 | 0.191689 |

## Closer Per Tick

- lstm: `219`
- xgboost: `25`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
