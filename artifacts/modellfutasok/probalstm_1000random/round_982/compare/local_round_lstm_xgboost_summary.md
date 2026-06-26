# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m3-inferno.csv`
- round_num: `15`
- rows: `135`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.256974 | 0.071196 | 0.301624 | 1.000000 | 0.743026 |
| xgboost | 0.296209 | 0.094856 | 0.358091 | 1.000000 | 0.703791 |

## Closer Per Tick

- lstm: `90`
- xgboost: `45`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
