# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `3`
- rows: `237`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.241504 | 0.093095 | 0.314662 | 0.890295 | 0.241504 |
| xgboost | 0.376959 | 0.178072 | 0.532064 | 0.763713 | 0.376959 |

## Closer Per Tick

- lstm: `219`
- xgboost: `18`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
