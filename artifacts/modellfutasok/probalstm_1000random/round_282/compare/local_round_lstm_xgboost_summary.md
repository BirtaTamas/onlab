# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `4`
- rows: `202`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.312610 | 0.106247 | 0.383033 | 1.000000 | 0.687390 |
| xgboost | 0.333099 | 0.122540 | 0.416965 | 0.970297 | 0.666901 |

## Closer Per Tick

- lstm: `126`
- xgboost: `76`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
