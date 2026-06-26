# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `5`
- rows: `251`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.345604 | 0.148330 | 0.458852 | 0.649402 | 0.345604 |
| xgboost | 0.345108 | 0.150829 | 0.462730 | 0.701195 | 0.345108 |

## Closer Per Tick

- lstm: `134`
- xgboost: `117`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
