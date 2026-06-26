# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9/vitality-vs-hotu-m2-dust2.csv`
- round_num: `15`
- rows: `138`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.575457 | 0.367894 | 0.995112 | 0.579710 | 0.424543 |
| xgboost | 0.556741 | 0.337752 | 0.879945 | 0.130435 | 0.443259 |

## Closer Per Tick

- lstm: `74`
- xgboost: `64`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
