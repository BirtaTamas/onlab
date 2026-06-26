# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `1`
- rows: `118`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.463353 | 0.236333 | 0.651811 | 0.576271 | 0.536647 |
| xgboost | 0.443912 | 0.222425 | 0.617356 | 0.516949 | 0.556088 |

## Closer Per Tick

- lstm: `54`
- xgboost: `64`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
