# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `3`
- rows: `122`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.323409 | 0.148571 | 0.437799 | 0.688525 | 0.323409 |
| xgboost | 0.325566 | 0.136086 | 0.424688 | 0.959016 | 0.325566 |

## Closer Per Tick

- lstm: `64`
- xgboost: `58`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
