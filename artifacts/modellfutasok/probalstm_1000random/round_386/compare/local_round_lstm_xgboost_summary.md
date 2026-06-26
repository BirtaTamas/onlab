# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-saw-bo3-hxORpk_jCtMpGRLo1Voi3p/furia-vs-saw-m2-dust2.csv`
- round_num: `14`
- rows: `118`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.016601 | 0.000385 | 0.016797 | 1.000000 | 0.016601 |
| xgboost | 0.017579 | 0.000373 | 0.017768 | 1.000000 | 0.017579 |

## Closer Per Tick

- lstm: `80`
- xgboost: `38`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
