# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `11`
- rows: `224`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.364223 | 0.184430 | 0.516357 | 0.491071 | 0.635777 |
| xgboost | 0.288522 | 0.126110 | 0.380937 | 1.000000 | 0.711478 |

## Closer Per Tick

- lstm: `0`
- xgboost: `224`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
