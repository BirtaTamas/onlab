# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `15`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.154902 | 0.042661 | 0.182403 | 1.000000 | 0.845098 |
| xgboost | 0.130648 | 0.041026 | 0.158777 | 1.000000 | 0.869352 |

## Closer Per Tick

- lstm: `37`
- xgboost: `193`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
