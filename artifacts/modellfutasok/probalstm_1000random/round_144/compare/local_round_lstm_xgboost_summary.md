# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-furia-vs-fluxo-bo3-cy88FeSpEinhT8XDRxQGHo/furia-vs-fluxo-m2-mirage.csv`
- round_num: `10`
- rows: `235`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.173287 | 0.077807 | 0.234071 | 0.765957 | 0.173287 |
| xgboost | 0.192347 | 0.075827 | 0.249056 | 0.778723 | 0.192347 |

## Closer Per Tick

- lstm: `159`
- xgboost: `76`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
