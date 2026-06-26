# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `9`
- rows: `234`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.090386 | 0.019711 | 0.102327 | 1.000000 | 0.090386 |
| xgboost | 0.141417 | 0.039434 | 0.166097 | 1.000000 | 0.141417 |

## Closer Per Tick

- lstm: `218`
- xgboost: `16`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
