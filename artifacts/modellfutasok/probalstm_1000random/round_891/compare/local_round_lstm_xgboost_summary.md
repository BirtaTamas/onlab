# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `14`
- rows: `157`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.677843 | 0.505074 | 1.341129 | 0.197452 | 0.322157 |
| xgboost | 0.596383 | 0.408913 | 1.038814 | 0.197452 | 0.403617 |

## Closer Per Tick

- lstm: `18`
- xgboost: `139`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
