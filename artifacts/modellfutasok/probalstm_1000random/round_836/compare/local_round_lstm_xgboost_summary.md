# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m2-nuke.csv`
- round_num: `6`
- rows: `151`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.246669 | 0.068062 | 0.289385 | 1.000000 | 0.753331 |
| xgboost | 0.183466 | 0.038670 | 0.206453 | 1.000000 | 0.816534 |

## Closer Per Tick

- lstm: `9`
- xgboost: `142`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
