# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m1-dust2.csv`
- round_num: `6`
- rows: `258`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.134520 | 0.027972 | 0.151833 | 1.000000 | 0.865480 |
| xgboost | 0.114013 | 0.021183 | 0.126736 | 1.000000 | 0.885987 |

## Closer Per Tick

- lstm: `77`
- xgboost: `181`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
