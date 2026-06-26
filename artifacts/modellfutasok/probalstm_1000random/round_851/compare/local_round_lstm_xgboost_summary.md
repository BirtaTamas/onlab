# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-pain-vs-housebets-bo3-SOezkQe1hszxnf1QDg0VUC/pain-vs-housebets-m1-dust2.csv`
- round_num: `2`
- rows: `210`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.691055 | 0.538984 | 1.493954 | 0.161905 | 0.308945 |
| xgboost | 0.580096 | 0.382142 | 0.961441 | 0.161905 | 0.419904 |

## Closer Per Tick

- lstm: `14`
- xgboost: `196`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
