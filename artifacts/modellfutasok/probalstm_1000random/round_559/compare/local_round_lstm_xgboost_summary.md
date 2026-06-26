# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `1`
- rows: `126`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.396205 | 0.198584 | 0.560239 | 0.730159 | 0.603795 |
| xgboost | 0.312213 | 0.136427 | 0.414460 | 0.912698 | 0.687787 |

## Closer Per Tick

- lstm: `2`
- xgboost: `124`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
