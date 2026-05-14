# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full\esl_pro_league_season_21\esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY\vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `12`
- rows: `176`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.234598 | 0.090753 | 0.304695 | 0.812500 | 0.234598 |
| xgboost | 0.297838 | 0.121997 | 0.392724 | 0.806818 | 0.297838 |

## Closer Per Tick

- lstm: `155`
- xgboost: `21`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
