# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `13`
- rows: `108`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.316943 | 0.130940 | 0.413072 | 1.000000 | 0.683057 |
| xgboost | 0.279107 | 0.106590 | 0.355510 | 1.000000 | 0.720893 |

## Closer Per Tick

- lstm: `18`
- xgboost: `90`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
