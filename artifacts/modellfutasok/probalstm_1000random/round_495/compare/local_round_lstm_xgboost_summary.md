# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `10`
- rows: `234`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.505332 | 0.298184 | 0.806759 | 0.487179 | 0.494668 |
| xgboost | 0.467439 | 0.247484 | 0.685324 | 0.564103 | 0.532561 |

## Closer Per Tick

- lstm: `77`
- xgboost: `157`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
