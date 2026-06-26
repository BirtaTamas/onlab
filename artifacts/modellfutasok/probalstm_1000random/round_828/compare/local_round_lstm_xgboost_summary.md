# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m2-nuke.csv`
- round_num: `7`
- rows: `172`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.206416 | 0.077916 | 0.260958 | 1.000000 | 0.793584 |
| xgboost | 0.204983 | 0.087161 | 0.268269 | 1.000000 | 0.795017 |

## Closer Per Tick

- lstm: `61`
- xgboost: `111`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
