# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-saw-bo3-hxORpk_jCtMpGRLo1Voi3p/furia-vs-saw-m2-dust2.csv`
- round_num: `13`
- rows: `129`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.211054 | 0.098874 | 0.286768 | 0.813953 | 0.211054 |
| xgboost | 0.213502 | 0.097093 | 0.288232 | 0.914729 | 0.213502 |

## Closer Per Tick

- lstm: `81`
- xgboost: `48`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
