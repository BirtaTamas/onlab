# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m3-ancient.csv`
- round_num: `12`
- rows: `178`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.425406 | 0.183787 | 0.558429 | 0.977528 | 0.574594 |
| xgboost | 0.473323 | 0.230117 | 0.653767 | 0.769663 | 0.526677 |

## Closer Per Tick

- lstm: `161`
- xgboost: `17`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
