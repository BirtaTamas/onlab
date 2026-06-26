# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-gentle-mates-bo3-EYv8hp-oY0glsojznK6Qby/legacy-vs-gentle-mates-m2-mirage.csv`
- round_num: `15`
- rows: `203`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.273364 | 0.092493 | 0.336975 | 0.965517 | 0.273364 |
| xgboost | 0.347528 | 0.138767 | 0.449275 | 0.684729 | 0.347528 |

## Closer Per Tick

- lstm: `195`
- xgboost: `8`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
