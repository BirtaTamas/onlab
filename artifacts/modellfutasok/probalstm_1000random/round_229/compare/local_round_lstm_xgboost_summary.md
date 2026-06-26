# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m2-mirage.csv`
- round_num: `14`
- rows: `128`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.392989 | 0.161262 | 0.507469 | 1.000000 | 0.607011 |
| xgboost | 0.276677 | 0.086081 | 0.333093 | 1.000000 | 0.723323 |

## Closer Per Tick

- lstm: `6`
- xgboost: `122`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
