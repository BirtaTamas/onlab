# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `3`
- rows: `174`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.056916 | 0.004428 | 0.059292 | 1.000000 | 0.943084 |
| xgboost | 0.020260 | 0.000505 | 0.020517 | 1.000000 | 0.979740 |

## Closer Per Tick

- lstm: `0`
- xgboost: `174`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
