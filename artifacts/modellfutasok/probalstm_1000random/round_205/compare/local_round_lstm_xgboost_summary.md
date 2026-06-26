# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv`
- round_num: `5`
- rows: `165`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.401556 | 0.187539 | 0.555263 | 0.860606 | 0.598444 |
| xgboost | 0.391287 | 0.177351 | 0.531409 | 0.860606 | 0.608713 |

## Closer Per Tick

- lstm: `57`
- xgboost: `108`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
