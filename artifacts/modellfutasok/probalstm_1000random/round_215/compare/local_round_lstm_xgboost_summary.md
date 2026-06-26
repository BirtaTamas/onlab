# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mibr-bo3-vjmAHfXA4PQfROTmirSCCF/vitality-vs-mibr-m2-inferno.csv`
- round_num: `10`
- rows: `263`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.150350 | 0.045093 | 0.181515 | 0.942966 | 0.150350 |
| xgboost | 0.194662 | 0.060285 | 0.233637 | 1.000000 | 0.194662 |

## Closer Per Tick

- lstm: `226`
- xgboost: `37`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
