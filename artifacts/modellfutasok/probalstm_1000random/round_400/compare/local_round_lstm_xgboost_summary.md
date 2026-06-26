# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-gamerlegion-vs-inner-circle-bo3-TOF4f6Uhtdi7Vqylk0QEY6/gamerlegion-vs-inner-circle-m1-ancient.csv`
- round_num: `11`
- rows: `163`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.695888 | 0.528851 | 1.692381 | 0.116564 | 0.304112 |
| xgboost | 0.591788 | 0.379477 | 1.042086 | 0.196319 | 0.408212 |

## Closer Per Tick

- lstm: `13`
- xgboost: `150`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
