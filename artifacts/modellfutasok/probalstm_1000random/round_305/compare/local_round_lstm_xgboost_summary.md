# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-inner-circle-bo3-YbhHiIk4CcU9clhSbtidF_/spirit-vs-inner-circle-m1-ancient.csv`
- round_num: `12`
- rows: `116`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.254353 | 0.140905 | 0.382450 | 0.715517 | 0.254353 |
| xgboost | 0.289807 | 0.135944 | 0.400839 | 0.681034 | 0.289807 |

## Closer Per Tick

- lstm: `77`
- xgboost: `39`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
