# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `7`
- rows: `104`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.013545 | 0.000353 | 0.013726 | 1.000000 | 0.013545 |
| xgboost | 0.020332 | 0.000782 | 0.020735 | 1.000000 | 0.020332 |

## Closer Per Tick

- lstm: `67`
- xgboost: `37`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
