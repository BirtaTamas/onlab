# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m2-ancient.csv`
- round_num: `10`
- rows: `136`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.261616 | 0.117805 | 0.357545 | 0.713235 | 0.261616 |
| xgboost | 0.213030 | 0.099245 | 0.296026 | 0.720588 | 0.213030 |

## Closer Per Tick

- lstm: `49`
- xgboost: `87`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
