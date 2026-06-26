# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `7`
- rows: `197`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.731035 | 0.570582 | 1.478840 | 0.081218 | 0.731035 |
| xgboost | 0.761305 | 0.595809 | 1.612671 | 0.086294 | 0.761305 |

## Closer Per Tick

- lstm: `100`
- xgboost: `97`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
