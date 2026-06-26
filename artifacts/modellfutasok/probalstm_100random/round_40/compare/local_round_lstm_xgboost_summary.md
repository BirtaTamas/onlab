# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m1-inferno.csv`
- round_num: `4`
- rows: `216`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.312204 | 0.183903 | 0.473863 | 0.504630 | 0.312204 |
| xgboost | 0.249596 | 0.115002 | 0.335433 | 1.000000 | 0.249596 |

## Closer Per Tick

- lstm: `100`
- xgboost: `116`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
