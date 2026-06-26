# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-m80-vs-rooster-bo3-GFAv4Fg83aXYKbsY0nLkP_/m80-vs-rooster-m2-inferno.csv`
- round_num: `7`
- rows: `138`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.216893 | 0.053848 | 0.249674 | 1.000000 | 0.783107 |
| xgboost | 0.213304 | 0.053839 | 0.246079 | 1.000000 | 0.786696 |

## Closer Per Tick

- lstm: `49`
- xgboost: `89`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
