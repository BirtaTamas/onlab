# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `1`
- rows: `152`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.338269 | 0.144570 | 0.445289 | 0.953947 | 0.661731 |
| xgboost | 0.260123 | 0.102556 | 0.335520 | 0.776316 | 0.739877 |

## Closer Per Tick

- lstm: `38`
- xgboost: `114`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
