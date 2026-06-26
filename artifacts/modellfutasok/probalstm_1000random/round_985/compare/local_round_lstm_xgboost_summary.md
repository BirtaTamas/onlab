# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `6`
- rows: `125`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.365179 | 0.170741 | 0.494473 | 0.760000 | 0.365179 |
| xgboost | 0.367522 | 0.151307 | 0.478332 | 0.848000 | 0.367522 |

## Closer Per Tick

- lstm: `56`
- xgboost: `69`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
