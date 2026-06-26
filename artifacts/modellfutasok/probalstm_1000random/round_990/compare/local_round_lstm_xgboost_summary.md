# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `9`
- rows: `134`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.257508 | 0.149554 | 0.392137 | 0.619403 | 0.257508 |
| xgboost | 0.217126 | 0.102177 | 0.296445 | 0.753731 | 0.217126 |

## Closer Per Tick

- lstm: `70`
- xgboost: `64`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
