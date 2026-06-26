# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `5`
- rows: `137`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 137 | 1.000 | 0.094493 | 0.140680 | -0.046187 | 131 | 6 | 0.985401 | 1.000000 |
| active/recent utility | 137 | 1.000 | 0.094493 | 0.140680 | -0.046187 | 131 | 6 | 0.985401 | 1.000000 |
| strong utility action | 117 | 0.854 | 0.102502 | 0.152116 | -0.049614 | 111 | 6 | 0.982906 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 107 | 0.781 | 0.099021 | 0.143721 | -0.044700 | 101 | 6 | 0.981308 | 1.000000 |
| recent utility last 5s | 10 | 0.073 | 0.139746 | 0.241943 | -0.102197 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 137 | 1.000 | 0.094493 | 0.140680 | -0.046187 | 131 | 6 | 0.985401 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `61.0s`, rows `107`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.0`, LSTM `0.0938`, XGBoost `0.2429`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.1076`, XGBoost `0.2359`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.1155`, XGBoost `0.2412`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.1213`, XGBoost `0.2412`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `28.0`, LSTM `0.1021`, XGBoost `0.2152`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.5`, LSTM `0.1319`, XGBoost `0.2430`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.0`, LSTM `0.1249`, XGBoost `0.2350`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.1058`, XGBoost `0.2152`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.1067`, XGBoost `0.2152`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.1078`, XGBoost `0.2152`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
