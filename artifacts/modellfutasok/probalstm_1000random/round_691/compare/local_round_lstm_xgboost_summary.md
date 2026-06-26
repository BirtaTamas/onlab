# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-faze-bo3-1414ljxN3FRmXv6-03KYFL/spirit-vs-faze-m2-mirage.csv`
- round_num: `12`
- rows: `116`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.283436 | 0.099596 | 0.350975 | 0.965517 | 0.716564 |
| xgboost | 0.297868 | 0.110764 | 0.373610 | 0.974138 | 0.702132 |

## Closer Per Tick

- lstm: `71`
- xgboost: `45`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
