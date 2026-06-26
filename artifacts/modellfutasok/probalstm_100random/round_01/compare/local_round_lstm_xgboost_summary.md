# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `20`
- rows: `153`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.604246 | 0.423178 | 1.070539 | 0.183007 | 0.604246 |
| xgboost | 0.604924 | 0.414271 | 1.057217 | 0.183007 | 0.604924 |

## Closer Per Tick

- lstm: `92`
- xgboost: `61`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
