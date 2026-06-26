# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m2-nuke.csv`
- round_num: `10`
- rows: `189`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.317662 | 0.130152 | 0.411853 | 0.989418 | 0.682338 |
| xgboost | 0.285617 | 0.111890 | 0.364104 | 0.989418 | 0.714383 |

## Closer Per Tick

- lstm: `32`
- xgboost: `157`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
