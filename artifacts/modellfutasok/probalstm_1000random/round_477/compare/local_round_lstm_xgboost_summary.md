# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `1`
- rows: `106`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.546817 | 0.364727 | 1.036169 | 0.311321 | 0.453183 |
| xgboost | 0.362761 | 0.179389 | 0.518782 | 0.858491 | 0.637239 |

## Closer Per Tick

- lstm: `0`
- xgboost: `106`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
