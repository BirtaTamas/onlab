# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-faze-bo3-1414ljxN3FRmXv6-03KYFL/spirit-vs-faze-m2-mirage.csv`
- round_num: `18`
- rows: `238`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.111393 | 0.036328 | 0.136293 | 1.000000 | 0.111393 |
| xgboost | 0.186211 | 0.068371 | 0.236419 | 1.000000 | 0.186211 |

## Closer Per Tick

- lstm: `238`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
