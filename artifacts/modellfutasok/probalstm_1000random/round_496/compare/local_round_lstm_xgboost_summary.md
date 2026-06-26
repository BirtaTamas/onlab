# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m1-inferno.csv`
- round_num: `8`
- rows: `173`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.365949 | 0.161812 | 0.487172 | 0.780347 | 0.634051 |
| xgboost | 0.388652 | 0.192233 | 0.542435 | 0.658960 | 0.611348 |

## Closer Per Tick

- lstm: `121`
- xgboost: `52`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
