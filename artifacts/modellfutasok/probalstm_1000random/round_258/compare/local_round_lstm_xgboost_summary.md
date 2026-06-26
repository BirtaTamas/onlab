# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `8`
- rows: `253`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.435760 | 0.201824 | 0.591032 | 0.667984 | 0.564240 |
| xgboost | 0.535690 | 0.306260 | 0.817260 | 0.557312 | 0.464310 |

## Closer Per Tick

- lstm: `236`
- xgboost: `17`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
