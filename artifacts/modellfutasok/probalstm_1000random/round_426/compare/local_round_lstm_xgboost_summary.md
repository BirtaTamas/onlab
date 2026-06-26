# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-falcons-vs-3dmax-bo3-XHM3Ovc8L9TfLFTYQFrGdT/falcons-vs-3dmax-m3-dust2.csv`
- round_num: `4`
- rows: `180`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.300694 | 0.105274 | 0.375948 | 0.861111 | 0.699306 |
| xgboost | 0.200274 | 0.050389 | 0.232808 | 1.000000 | 0.799726 |

## Closer Per Tick

- lstm: `0`
- xgboost: `180`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
