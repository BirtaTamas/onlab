# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `13`
- rows: `207`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.378815 | 0.168386 | 0.503889 | 0.743961 | 0.378815 |
| xgboost | 0.437846 | 0.218001 | 0.610221 | 0.502415 | 0.437846 |

## Closer Per Tick

- lstm: `189`
- xgboost: `18`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
