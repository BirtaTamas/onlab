# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-b8-bo3--nzkpOWiS4qFgkFOwM8Hun/legacy-vs-b8-m2-ancient.csv`
- round_num: `19`
- rows: `117`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.543944 | 0.322606 | 0.855134 | 0.418803 | 0.543944 |
| xgboost | 0.520043 | 0.304113 | 0.811875 | 0.418803 | 0.520043 |

## Closer Per Tick

- lstm: `50`
- xgboost: `67`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
