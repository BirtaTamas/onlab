# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `7`
- rows: `206`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.637934 | 0.435834 | 1.217688 | 0.131068 | 0.362066 |
| xgboost | 0.577244 | 0.355010 | 0.937647 | 0.169903 | 0.422756 |

## Closer Per Tick

- lstm: `73`
- xgboost: `133`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
