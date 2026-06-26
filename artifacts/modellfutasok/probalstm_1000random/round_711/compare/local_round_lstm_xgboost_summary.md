# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `10`
- rows: `247`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.245578 | 0.102283 | 0.318581 | 1.000000 | 0.245578 |
| xgboost | 0.285083 | 0.134527 | 0.387282 | 0.939271 | 0.285083 |

## Closer Per Tick

- lstm: `235`
- xgboost: `12`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
