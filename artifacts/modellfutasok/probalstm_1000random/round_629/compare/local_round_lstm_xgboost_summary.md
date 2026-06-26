# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `17`
- rows: `164`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.151360 | 0.040574 | 0.177298 | 1.000000 | 0.848640 |
| xgboost | 0.158292 | 0.048470 | 0.189965 | 1.000000 | 0.841708 |

## Closer Per Tick

- lstm: `65`
- xgboost: `99`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
