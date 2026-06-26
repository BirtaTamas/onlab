# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-b8-vs-hotu-bo3-tmCfOETKzYqjV6vSvNp3-F/b8-vs-hotu-m3-ancient.csv`
- round_num: `2`
- rows: `112`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.049563 | 0.003426 | 0.051378 | 1.000000 | 0.950437 |
| xgboost | 0.015735 | 0.000338 | 0.015908 | 1.000000 | 0.984265 |

## Closer Per Tick

- lstm: `0`
- xgboost: `112`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
