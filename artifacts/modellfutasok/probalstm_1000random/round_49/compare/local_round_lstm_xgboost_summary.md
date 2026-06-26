# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-gentle-mates-bo3-EYv8hp-oY0glsojznK6Qby/legacy-vs-gentle-mates-m2-mirage.csv`
- round_num: `11`
- rows: `207`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.352898 | 0.138430 | 0.451200 | 0.937198 | 0.647102 |
| xgboost | 0.282832 | 0.094630 | 0.347586 | 1.000000 | 0.717168 |

## Closer Per Tick

- lstm: `4`
- xgboost: `203`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
