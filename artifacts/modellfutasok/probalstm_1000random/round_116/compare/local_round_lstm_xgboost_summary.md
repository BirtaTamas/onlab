# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `13`
- rows: `192`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.627042 | 0.454690 | 1.464605 | 0.520833 | 0.372958 |
| xgboost | 0.590037 | 0.422327 | 1.315164 | 0.531250 | 0.409963 |

## Closer Per Tick

- lstm: `35`
- xgboost: `157`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
