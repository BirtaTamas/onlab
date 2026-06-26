# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-faze-bo3-ZgdBOa3Yi0KCkwa_Ap1ef3/aurora-vs-faze-m2-train.csv`
- round_num: `15`
- rows: `190`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.288162 | 0.107763 | 0.366404 | 0.952632 | 0.711838 |
| xgboost | 0.269215 | 0.110904 | 0.361414 | 0.952632 | 0.730785 |

## Closer Per Tick

- lstm: `61`
- xgboost: `129`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
