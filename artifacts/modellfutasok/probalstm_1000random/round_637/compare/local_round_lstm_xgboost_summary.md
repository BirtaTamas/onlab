# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv`
- round_num: `2`
- rows: `198`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.063225 | 0.007879 | 0.067831 | 1.000000 | 0.936775 |
| xgboost | 0.024835 | 0.001320 | 0.025536 | 1.000000 | 0.975165 |

## Closer Per Tick

- lstm: `0`
- xgboost: `198`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
