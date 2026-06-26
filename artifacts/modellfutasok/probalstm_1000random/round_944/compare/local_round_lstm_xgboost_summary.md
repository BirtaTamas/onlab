# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m3-ancient.csv`
- round_num: `5`
- rows: `119`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.317668 | 0.135160 | 0.421948 | 0.689076 | 0.682332 |
| xgboost | 0.245677 | 0.083692 | 0.303867 | 1.000000 | 0.754323 |

## Closer Per Tick

- lstm: `4`
- xgboost: `115`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
