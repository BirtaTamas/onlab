# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `11`
- rows: `234`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.224459 | 0.113470 | 0.316567 | 0.662393 | 0.224459 |
| xgboost | 0.188702 | 0.084465 | 0.253341 | 0.897436 | 0.188702 |

## Closer Per Tick

- lstm: `68`
- xgboost: `166`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
