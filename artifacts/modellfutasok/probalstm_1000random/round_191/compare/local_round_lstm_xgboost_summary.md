# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-3dmax-bo3-Dgk7HiwYvj5CMwMpEHLxHJ/heroic-vs-3dmax-m1-nuke.csv`
- round_num: `10`
- rows: `209`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.453543 | 0.295434 | 0.756600 | 0.406699 | 0.453543 |
| xgboost | 0.472519 | 0.312987 | 0.804677 | 0.440191 | 0.472519 |

## Closer Per Tick

- lstm: `141`
- xgboost: `68`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
