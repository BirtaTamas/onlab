# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `4`
- rows: `266`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.338132 | 0.178676 | 0.487345 | 0.548872 | 0.338132 |
| xgboost | 0.388159 | 0.235682 | 0.626320 | 0.582707 | 0.388159 |

## Closer Per Tick

- lstm: `235`
- xgboost: `31`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
