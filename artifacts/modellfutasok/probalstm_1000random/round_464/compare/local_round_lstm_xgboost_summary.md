# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-b8-bo3-rUWlZLFFckLiQv1C1wSlHb/g2-vs-b8-m3-ancient.csv`
- round_num: `6`
- rows: `217`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.308347 | 0.109067 | 0.382638 | 1.000000 | 0.691653 |
| xgboost | 0.354828 | 0.137517 | 0.450426 | 1.000000 | 0.645172 |

## Closer Per Tick

- lstm: `167`
- xgboost: `50`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
