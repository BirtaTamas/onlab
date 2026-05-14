# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full\esl_pro_league_season_22\esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3\the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `7`
- rows: `180`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.244666 | 0.100392 | 0.318801 | 0.850000 | 0.244666 |
| xgboost | 0.368705 | 0.166689 | 0.499529 | 0.594444 | 0.368705 |

## Closer Per Tick

- lstm: `178`
- xgboost: `2`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
