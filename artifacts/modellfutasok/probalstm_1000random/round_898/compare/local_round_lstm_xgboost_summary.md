# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-furia-vs-fluxo-bo3-cy88FeSpEinhT8XDRxQGHo/furia-vs-fluxo-m2-mirage.csv`
- round_num: `15`
- rows: `187`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.265454 | 0.086233 | 0.324381 | 0.951872 | 0.734546 |
| xgboost | 0.255398 | 0.082490 | 0.312532 | 0.882353 | 0.744602 |

## Closer Per Tick

- lstm: `89`
- xgboost: `98`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
