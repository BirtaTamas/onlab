# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m1-inferno.csv`
- round_num: `10`
- rows: `205`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.196278 | 0.062337 | 0.238350 | 1.000000 | 0.803722 |
| xgboost | 0.205776 | 0.091550 | 0.275070 | 0.980488 | 0.794224 |

## Closer Per Tick

- lstm: `78`
- xgboost: `127`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
