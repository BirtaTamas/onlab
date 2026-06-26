# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-fluxo-bo3-sWQe-jgKNP3vaioXQrjxgB/astralis-vs-fluxo-m3-nuke.csv`
- round_num: `13`
- rows: `157`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.272199 | 0.124098 | 0.367448 | 0.891720 | 0.727801 |
| xgboost | 0.228734 | 0.103369 | 0.307562 | 0.904459 | 0.771266 |

## Closer Per Tick

- lstm: `20`
- xgboost: `137`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
