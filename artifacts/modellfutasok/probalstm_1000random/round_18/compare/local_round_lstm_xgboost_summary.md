# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-3dmax-bo3-Oe166BQltZjvHlE8qlepgF/furia-vs-3dmax-m1-nuke.csv`
- round_num: `6`
- rows: `168`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.366889 | 0.140934 | 0.464750 | 1.000000 | 0.633111 |
| xgboost | 0.294059 | 0.097006 | 0.359448 | 1.000000 | 0.705941 |

## Closer Per Tick

- lstm: `10`
- xgboost: `158`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
