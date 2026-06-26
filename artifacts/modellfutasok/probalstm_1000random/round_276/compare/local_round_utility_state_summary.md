# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `2`
- rows: `120`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 120 | 1.000 | 0.760486 | 0.790640 | -0.030154 | 15 | 105 | 0.975000 | 0.991667 |
| active/recent utility | 120 | 1.000 | 0.760486 | 0.790640 | -0.030154 | 15 | 105 | 0.975000 | 0.991667 |
| strong utility action | 92 | 0.767 | 0.729090 | 0.765233 | -0.036142 | 11 | 81 | 0.967391 | 0.989130 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 88 | 0.733 | 0.734264 | 0.772064 | -0.037800 | 9 | 79 | 0.965909 | 0.988636 |
| recent utility last 5s | 10 | 0.083 | 0.618855 | 0.614240 | 0.004615 | 7 | 3 | 1.000000 | 1.000000 |
| flash effect present | 120 | 1.000 | 0.760486 | 0.790640 | -0.030154 | 15 | 105 | 0.975000 | 0.991667 |

## Active Smoke/Inferno Intervals

- `6.0s` - `49.5s`, rows `88`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.0`, LSTM `0.7627`, XGBoost `0.8938`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.4295`, XGBoost `0.5223`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5007`, XGBoost `0.5935`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.8070`, XGBoost `0.8987`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.8108`, XGBoost `0.8971`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5098`, XGBoost `0.5935`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5119`, XGBoost `0.5844`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.8993`, XGBoost `0.9700`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.9039`, XGBoost `0.9700`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5192`, XGBoost `0.5842`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
