# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3/the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `3`
- rows: `145`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.162632 | 0.063525 | 0.207688 | 1.000000 | 0.162632 |
| xgboost | 0.221342 | 0.094577 | 0.291741 | 0.986207 | 0.221342 |

## Closer Per Tick

- lstm: `139`
- xgboost: `6`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
