# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `4`
- rows: `267`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.606346 | 0.427772 | 1.092946 | 0.217228 | 0.393654 |
| xgboost | 0.512024 | 0.307955 | 0.795043 | 0.217228 | 0.487976 |

## Closer Per Tick

- lstm: `45`
- xgboost: `222`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
