# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `18`
- rows: `146`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.755110 | 0.607796 | 1.728281 | 0.157534 | 0.244890 |
| xgboost | 0.683560 | 0.525201 | 1.396146 | 0.164384 | 0.316440 |

## Closer Per Tick

- lstm: `10`
- xgboost: `136`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
