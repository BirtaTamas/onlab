# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9/vitality-vs-hotu-m2-dust2.csv`
- round_num: `5`
- rows: `162`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.008677 | 0.000140 | 0.008748 | 1.000000 | 0.008677 |
| xgboost | 0.040943 | 0.002553 | 0.042277 | 1.000000 | 0.040943 |

## Closer Per Tick

- lstm: `137`
- xgboost: `25`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
