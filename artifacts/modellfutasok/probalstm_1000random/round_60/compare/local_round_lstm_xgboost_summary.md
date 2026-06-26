# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m3-ancient.csv`
- round_num: `18`
- rows: `245`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.326257 | 0.139512 | 0.444665 | 0.914286 | 0.326257 |
| xgboost | 0.450278 | 0.221170 | 0.629948 | 0.587755 | 0.450278 |

## Closer Per Tick

- lstm: `209`
- xgboost: `36`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
