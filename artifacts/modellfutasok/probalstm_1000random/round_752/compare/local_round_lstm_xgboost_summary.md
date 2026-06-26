# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `10`
- rows: `176`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.316802 | 0.114932 | 0.395199 | 1.000000 | 0.683198 |
| xgboost | 0.303561 | 0.112325 | 0.381451 | 1.000000 | 0.696439 |

## Closer Per Tick

- lstm: `79`
- xgboost: `97`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
