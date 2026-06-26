# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `1`
- rows: `157`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.311414 | 0.111202 | 0.388688 | 0.974522 | 0.688586 |
| xgboost | 0.211787 | 0.063986 | 0.255195 | 1.000000 | 0.788213 |

## Closer Per Tick

- lstm: `0`
- xgboost: `157`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
