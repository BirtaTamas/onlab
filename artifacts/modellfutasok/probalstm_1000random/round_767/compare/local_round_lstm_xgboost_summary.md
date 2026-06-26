# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `13`
- rows: `115`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.481997 | 0.241992 | 0.670660 | 0.330435 | 0.518003 |
| xgboost | 0.447673 | 0.213792 | 0.609740 | 0.765217 | 0.552327 |

## Closer Per Tick

- lstm: `25`
- xgboost: `90`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
