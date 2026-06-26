# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `18`
- rows: `228`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.520681 | 0.274733 | 0.743078 | 0.368421 | 0.520681 |
| xgboost | 0.387592 | 0.167638 | 0.514045 | 0.688596 | 0.387592 |

## Closer Per Tick

- lstm: `0`
- xgboost: `228`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
