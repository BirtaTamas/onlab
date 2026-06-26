# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-astralis-bo3-EH-le-_LuObR5nGefXVoZY/aurora-vs-astralis-m3-overpass.csv`
- round_num: `2`
- rows: `224`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.360852 | 0.146752 | 0.466624 | 0.977679 | 0.639148 |
| xgboost | 0.259940 | 0.081497 | 0.313419 | 1.000000 | 0.740060 |

## Closer Per Tick

- lstm: `0`
- xgboost: `224`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
