# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m1-inferno.csv`
- round_num: `6`
- rows: `195`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.314800 | 0.108150 | 0.387052 | 1.000000 | 0.685200 |
| xgboost | 0.361582 | 0.144654 | 0.465083 | 0.800000 | 0.638418 |

## Closer Per Tick

- lstm: `172`
- xgboost: `23`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
