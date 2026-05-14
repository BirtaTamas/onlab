# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full\esl_pro_league_season_21\esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY\vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `6`
- rows: `144`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.664391 | 0.514189 | 1.375004 | 0.166667 | 0.335609 |
| xgboost | 0.533656 | 0.335599 | 0.855990 | 0.215278 | 0.466344 |

## Closer Per Tick

- lstm: `0`
- xgboost: `144`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
