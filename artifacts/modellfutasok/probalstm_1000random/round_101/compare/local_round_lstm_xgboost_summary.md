# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `3`
- rows: `179`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.044965 | 0.002651 | 0.046359 | 1.000000 | 0.044965 |
| xgboost | 0.065321 | 0.004872 | 0.067909 | 1.000000 | 0.065321 |

## Closer Per Tick

- lstm: `168`
- xgboost: `11`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
