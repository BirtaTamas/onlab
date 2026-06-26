# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-gentle-mates-vs-aurora-bo3-gDH2lDrlT5ROvKI-0e6nmI/gentle-mates-vs-aurora-m1-nuke.csv`
- round_num: `15`
- rows: `203`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.217220 | 0.084700 | 0.278905 | 0.852217 | 0.217220 |
| xgboost | 0.231566 | 0.092553 | 0.299424 | 1.000000 | 0.231566 |

## Closer Per Tick

- lstm: `132`
- xgboost: `71`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
