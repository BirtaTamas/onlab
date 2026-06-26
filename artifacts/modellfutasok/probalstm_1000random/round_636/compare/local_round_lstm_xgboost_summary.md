# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `19`
- rows: `211`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.018967 | 0.000473 | 0.019207 | 1.000000 | 0.018967 |
| xgboost | 0.058588 | 0.004125 | 0.060755 | 1.000000 | 0.058588 |

## Closer Per Tick

- lstm: `202`
- xgboost: `9`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
