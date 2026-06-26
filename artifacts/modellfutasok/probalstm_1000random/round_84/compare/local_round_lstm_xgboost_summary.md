# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-3dmax-bo3-Dgk7HiwYvj5CMwMpEHLxHJ/heroic-vs-3dmax-m1-nuke.csv`
- round_num: `13`
- rows: `161`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.736219 | 0.578481 | 1.668020 | 0.062112 | 0.736219 |
| xgboost | 0.815069 | 0.695385 | 2.253124 | 0.074534 | 0.815069 |

## Closer Per Tick

- lstm: `135`
- xgboost: `26`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
