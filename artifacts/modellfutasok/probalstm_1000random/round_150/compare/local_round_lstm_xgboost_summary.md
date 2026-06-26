# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-nrg-vs-fluxo-bo3-aFv0UX6WO0txoeY8N630nT/nrg-vs-fluxo-m1-nuke.csv`
- round_num: `20`
- rows: `194`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.206004 | 0.052994 | 0.239829 | 1.000000 | 0.793996 |
| xgboost | 0.136134 | 0.027144 | 0.152565 | 1.000000 | 0.863866 |

## Closer Per Tick

- lstm: `1`
- xgboost: `193`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
