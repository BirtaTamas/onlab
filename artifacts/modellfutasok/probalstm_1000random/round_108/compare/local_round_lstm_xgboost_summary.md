# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `10`
- rows: `160`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.024073 | 0.001336 | 0.024783 | 1.000000 | 0.024073 |
| xgboost | 0.048523 | 0.005805 | 0.051925 | 1.000000 | 0.048523 |

## Closer Per Tick

- lstm: `158`
- xgboost: `2`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
