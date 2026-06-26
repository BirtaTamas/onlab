# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `2`
- rows: `120`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.239514 | 0.090358 | 0.303887 | 0.975000 | 0.760486 |
| xgboost | 0.209360 | 0.077857 | 0.263353 | 0.991667 | 0.790640 |

## Closer Per Tick

- lstm: `15`
- xgboost: `105`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
