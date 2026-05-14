# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full\esl_pro_league_season_22\esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9\vitality-vs-hotu-m2-dust2.csv`
- round_num: `3`
- rows: `126`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.144102 | 0.057544 | 0.188423 | 0.857143 | 0.144102 |
| xgboost | 0.128175 | 0.040502 | 0.156515 | 1.000000 | 0.128175 |

## Closer Per Tick

- lstm: `77`
- xgboost: `49`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
