# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m3-anubis.csv`
- round_num: `2`
- rows: `151`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.013384 | 0.000259 | 0.013516 | 1.000000 | 0.013384 |
| xgboost | 0.054478 | 0.003230 | 0.056161 | 1.000000 | 0.054478 |

## Closer Per Tick

- lstm: `151`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
