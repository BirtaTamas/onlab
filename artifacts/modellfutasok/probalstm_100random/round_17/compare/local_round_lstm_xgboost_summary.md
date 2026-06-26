# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv`
- round_num: `6`
- rows: `253`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.388663 | 0.203816 | 0.576853 | 0.695652 | 0.611337 |
| xgboost | 0.283966 | 0.099992 | 0.354175 | 0.897233 | 0.716034 |

## Closer Per Tick

- lstm: `65`
- xgboost: `188`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
