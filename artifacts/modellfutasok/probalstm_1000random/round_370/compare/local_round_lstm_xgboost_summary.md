# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `13`
- rows: `144`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.383253 | 0.186583 | 0.529215 | 0.770833 | 0.616747 |
| xgboost | 0.360223 | 0.174327 | 0.493857 | 0.777778 | 0.639777 |

## Closer Per Tick

- lstm: `26`
- xgboost: `118`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
