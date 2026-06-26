# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `6`
- rows: `244`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.686612 | 0.526987 | 1.471577 | 0.098361 | 0.313388 |
| xgboost | 0.550068 | 0.349526 | 0.919495 | 0.188525 | 0.449932 |

## Closer Per Tick

- lstm: `0`
- xgboost: `244`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
