# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `8`
- rows: `132`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 132 | 1.000 | 0.110462 | 0.147172 | -0.036710 | 102 | 30 | 1.000000 | 1.000000 |
| active/recent utility | 132 | 1.000 | 0.110462 | 0.147172 | -0.036710 | 102 | 30 | 1.000000 | 1.000000 |
| strong utility action | 125 | 0.947 | 0.110374 | 0.145215 | -0.034842 | 96 | 29 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 111 | 0.841 | 0.109718 | 0.131316 | -0.021598 | 82 | 29 | 1.000000 | 1.000000 |
| recent utility last 5s | 14 | 0.106 | 0.115573 | 0.255419 | -0.139846 | 14 | 0 | 1.000000 | 1.000000 |
| flash effect present | 132 | 1.000 | 0.110462 | 0.147172 | -0.036710 | 102 | 30 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `64.5s`, rows `111`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `3.0`, LSTM `0.0887`, XGBoost `0.2554`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `3.5`, LSTM `0.0987`, XGBoost `0.2554`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `2.5`, LSTM `0.1078`, XGBoost `0.2557`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.5`, LSTM `0.1087`, XGBoost `0.2564`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.1097`, XGBoost `0.2554`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.1098`, XGBoost `0.2554`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `2.0`, LSTM `0.1146`, XGBoost `0.2547`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.1196`, XGBoost `0.2554`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `4.5`, LSTM `0.1203`, XGBoost `0.2554`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `6.5`, LSTM `0.1210`, XGBoost `0.2554`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
