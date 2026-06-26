# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `11`
- rows: `192`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.347602 | 0.139005 | 0.446402 | 0.973958 | 0.652398 |
| xgboost | 0.365463 | 0.153416 | 0.476199 | 1.000000 | 0.634537 |

## Closer Per Tick

- lstm: `120`
- xgboost: `72`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
