# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m3-anubis.csv`
- round_num: `6`
- rows: `195`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.244477 | 0.124998 | 0.344976 | 0.656410 | 0.244477 |
| xgboost | 0.253583 | 0.119694 | 0.347460 | 0.712821 | 0.253583 |

## Closer Per Tick

- lstm: `136`
- xgboost: `59`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
