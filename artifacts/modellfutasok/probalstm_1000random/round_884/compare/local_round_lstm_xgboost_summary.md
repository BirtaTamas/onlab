# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `10`
- rows: `192`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.317080 | 0.152406 | 0.438100 | 0.713542 | 0.317080 |
| xgboost | 0.468325 | 0.266824 | 0.733044 | 0.489583 | 0.468325 |

## Closer Per Tick

- lstm: `168`
- xgboost: `24`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
