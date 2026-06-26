# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-3dmax-bo3-Oe166BQltZjvHlE8qlepgF/furia-vs-3dmax-m1-nuke.csv`
- round_num: `9`
- rows: `212`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.121091 | 0.031084 | 0.141670 | 1.000000 | 0.878909 |
| xgboost | 0.101124 | 0.029883 | 0.121217 | 1.000000 | 0.898876 |

## Closer Per Tick

- lstm: `36`
- xgboost: `176`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
