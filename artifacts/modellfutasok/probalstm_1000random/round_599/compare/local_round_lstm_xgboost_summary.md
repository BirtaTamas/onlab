# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `8`
- rows: `285`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.205294 | 0.067386 | 0.253525 | 0.908772 | 0.794706 |
| xgboost | 0.185697 | 0.054924 | 0.222646 | 1.000000 | 0.814303 |

## Closer Per Tick

- lstm: `69`
- xgboost: `216`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
