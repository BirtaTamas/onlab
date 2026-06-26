# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `9`
- rows: `202`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.368310 | 0.148293 | 0.473712 | 1.000000 | 0.631690 |
| xgboost | 0.388979 | 0.164016 | 0.507609 | 0.985149 | 0.611021 |

## Closer Per Tick

- lstm: `155`
- xgboost: `47`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
