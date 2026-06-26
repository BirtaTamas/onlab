# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `10`
- rows: `205`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.468455 | 0.239807 | 0.676102 | 0.673171 | 0.531545 |
| xgboost | 0.407212 | 0.186702 | 0.554363 | 0.692683 | 0.592788 |

## Closer Per Tick

- lstm: `23`
- xgboost: `182`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
