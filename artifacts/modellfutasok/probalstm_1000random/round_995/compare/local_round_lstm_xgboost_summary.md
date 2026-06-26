# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m1-inferno.csv`
- round_num: `6`
- rows: `174`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.240787 | 0.066532 | 0.282145 | 1.000000 | 0.759213 |
| xgboost | 0.237706 | 0.064905 | 0.277750 | 1.000000 | 0.762294 |

## Closer Per Tick

- lstm: `54`
- xgboost: `120`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
