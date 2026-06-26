# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-pain-vs-housebets-bo3-SOezkQe1hszxnf1QDg0VUC/pain-vs-housebets-m1-dust2.csv`
- round_num: `12`
- rows: `189`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.346654 | 0.177012 | 0.495907 | 0.656085 | 0.346654 |
| xgboost | 0.281547 | 0.110710 | 0.363483 | 0.878307 | 0.281547 |

## Closer Per Tick

- lstm: `64`
- xgboost: `125`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
