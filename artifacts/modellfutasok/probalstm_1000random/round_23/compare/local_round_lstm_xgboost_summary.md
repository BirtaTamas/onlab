# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-pain-bo3-zcuZjSa9VUSMkJoK5k8I3c/gamerlegion-vs-pain-m3-mirage.csv`
- round_num: `5`
- rows: `208`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.127470 | 0.050077 | 0.164249 | 0.975962 | 0.127470 |
| xgboost | 0.150383 | 0.060261 | 0.196168 | 0.817308 | 0.150383 |

## Closer Per Tick

- lstm: `184`
- xgboost: `24`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
