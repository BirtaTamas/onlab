# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `5`
- rows: `178`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.204788 | 0.049154 | 0.234567 | 1.000000 | 0.795212 |
| xgboost | 0.134706 | 0.021813 | 0.146997 | 1.000000 | 0.865294 |

## Closer Per Tick

- lstm: `0`
- xgboost: `178`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
