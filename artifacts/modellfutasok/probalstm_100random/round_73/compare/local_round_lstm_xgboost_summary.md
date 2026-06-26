# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-m80-vs-flyquest-bo3-ji2oWF2IQJDeDBfGP8d4J9/m80-vs-flyquest-m2-dust2.csv`
- round_num: `21`
- rows: `176`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.531147 | 0.327309 | 0.851343 | 0.215909 | 0.531147 |
| xgboost | 0.584899 | 0.377773 | 0.961433 | 0.159091 | 0.584899 |

## Closer Per Tick

- lstm: `153`
- xgboost: `23`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
