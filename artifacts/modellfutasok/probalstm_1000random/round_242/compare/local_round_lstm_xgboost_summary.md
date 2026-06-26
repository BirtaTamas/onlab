# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `4`
- rows: `199`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.497636 | 0.322973 | 0.885334 | 0.457286 | 0.502364 |
| xgboost | 0.443095 | 0.294333 | 0.778662 | 0.391960 | 0.556905 |

## Closer Per Tick

- lstm: `48`
- xgboost: `151`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
