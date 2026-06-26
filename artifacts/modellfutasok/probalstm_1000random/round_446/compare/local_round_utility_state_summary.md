# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-inner-circle-bo3-YbhHiIk4CcU9clhSbtidF_/spirit-vs-inner-circle-m1-ancient.csv`
- round_num: `11`
- rows: `189`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 189 | 1.000 | 0.659965 | 0.714116 | -0.054151 | 31 | 158 | 0.735450 | 0.783069 |
| active/recent utility | 189 | 1.000 | 0.659965 | 0.714116 | -0.054151 | 31 | 158 | 0.735450 | 0.783069 |
| strong utility action | 140 | 0.741 | 0.633146 | 0.684608 | -0.051462 | 28 | 112 | 0.721429 | 0.785714 |
| utility damage | 16 | 0.085 | 0.583519 | 0.570086 | 0.013432 | 10 | 6 | 1.000000 | 0.937500 |
| active smoke/inferno | 132 | 0.698 | 0.639779 | 0.695339 | -0.055559 | 20 | 112 | 0.704545 | 0.780303 |
| recent utility last 5s | 20 | 0.106 | 0.538118 | 0.544230 | -0.006112 | 14 | 6 | 0.850000 | 0.900000 |
| flash effect present | 189 | 1.000 | 0.659965 | 0.714116 | -0.054151 | 31 | 158 | 0.735450 | 0.783069 |

## Active Smoke/Inferno Intervals

- `5.5s` - `33.0s`, rows `56`
- `39.0s` - `69.5s`, rows `62`
- `83.0s` - `89.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.5`, LSTM `0.1016`, XGBoost `0.3011`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.1058`, XGBoost `0.3033`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.1065`, XGBoost `0.3033`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.1124`, XGBoost `0.3069`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.1129`, XGBoost `0.3069`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.3067`, XGBoost `0.4928`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.1180`, XGBoost `0.3020`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.1235`, XGBoost `0.3018`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1343`, XGBoost `0.3069`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.1444`, XGBoost `0.3161`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
