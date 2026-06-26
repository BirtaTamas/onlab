# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m2-dust2.csv`
- round_num: `10`
- rows: `297`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.237233 | 0.083261 | 0.292884 | 1.000000 | 0.237233 |
| xgboost | 0.344281 | 0.161470 | 0.467056 | 0.794613 | 0.344281 |

## Closer Per Tick

- lstm: `297`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
