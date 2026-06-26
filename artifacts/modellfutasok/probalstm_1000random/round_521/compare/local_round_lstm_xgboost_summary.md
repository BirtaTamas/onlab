# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `16`
- rows: `170`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.158836 | 0.039154 | 0.184100 | 1.000000 | 0.841164 |
| xgboost | 0.152219 | 0.046703 | 0.184960 | 1.000000 | 0.847781 |

## Closer Per Tick

- lstm: `56`
- xgboost: `114`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
