# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `5`
- rows: `143`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.175488 | 0.051692 | 0.208937 | 1.000000 | 0.824512 |
| xgboost | 0.117316 | 0.025016 | 0.132083 | 1.000000 | 0.882684 |

## Closer Per Tick

- lstm: `3`
- xgboost: `140`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
