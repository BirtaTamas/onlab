# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `2`
- rows: `176`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.076416 | 0.006202 | 0.079708 | 1.000000 | 0.923584 |
| xgboost | 0.020121 | 0.000414 | 0.020331 | 1.000000 | 0.979879 |

## Closer Per Tick

- lstm: `0`
- xgboost: `176`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
