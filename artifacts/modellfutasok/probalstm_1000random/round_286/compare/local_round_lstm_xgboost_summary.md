# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-b8-vs-hotu-bo3-tmCfOETKzYqjV6vSvNp3-F/b8-vs-hotu-m3-ancient.csv`
- round_num: `6`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.084554 | 0.015959 | 0.094174 | 1.000000 | 0.915446 |
| xgboost | 0.066105 | 0.017204 | 0.077428 | 1.000000 | 0.933895 |

## Closer Per Tick

- lstm: `28`
- xgboost: `202`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
