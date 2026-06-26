# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m3-ancient.csv`
- round_num: `16`
- rows: `164`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.104009 | 0.047280 | 0.140039 | 0.945122 | 0.104009 |
| xgboost | 0.118368 | 0.052648 | 0.159466 | 0.853659 | 0.118368 |

## Closer Per Tick

- lstm: `153`
- xgboost: `11`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
