# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-gentle-mates-vs-aurora-bo3-gDH2lDrlT5ROvKI-0e6nmI/gentle-mates-vs-aurora-m1-nuke.csv`
- round_num: `5`
- rows: `143`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.327455 | 0.126092 | 0.414813 | 1.000000 | 0.672545 |
| xgboost | 0.341820 | 0.146373 | 0.448451 | 1.000000 | 0.658180 |

## Closer Per Tick

- lstm: `94`
- xgboost: `49`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
