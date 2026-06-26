# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m2-ancient.csv`
- round_num: `18`
- rows: `113`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.206671 | 0.115768 | 0.305898 | 0.734513 | 0.206671 |
| xgboost | 0.188452 | 0.096251 | 0.267550 | 0.734513 | 0.188452 |

## Closer Per Tick

- lstm: `73`
- xgboost: `40`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
