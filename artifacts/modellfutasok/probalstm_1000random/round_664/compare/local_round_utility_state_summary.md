# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `7`
- rows: `107`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 107 | 1.000 | 0.090868 | 0.108273 | -0.017405 | 102 | 5 | 1.000000 | 0.934579 |
| active/recent utility | 107 | 1.000 | 0.090868 | 0.108273 | -0.017405 | 102 | 5 | 1.000000 | 0.934579 |
| strong utility action | 91 | 0.850 | 0.064897 | 0.080396 | -0.015499 | 86 | 5 | 1.000000 | 0.956044 |
| utility damage | 10 | 0.093 | 0.302123 | 0.321842 | -0.019718 | 7 | 3 | 1.000000 | 0.600000 |
| active smoke/inferno | 81 | 0.757 | 0.035609 | 0.050588 | -0.014978 | 79 | 2 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 107 | 1.000 | 0.090868 | 0.108273 | -0.017405 | 102 | 5 | 1.000000 | 0.934579 |

## Active Smoke/Inferno Intervals

- `9.5s` - `49.5s`, rows `81`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `6.0`, LSTM `0.3318`, XGBoost `0.2024`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0979`, XGBoost `0.2024`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0994`, XGBoost `0.2024`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1204`, XGBoost `0.2061`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1234`, XGBoost `0.2061`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.1311`, XGBoost `0.2024`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.1636`, XGBoost `0.2217`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.1619`, XGBoost `0.2193`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `4.5`, LSTM `0.4477`, XGBoost `0.5012`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `4.0`, LSTM `0.4477`, XGBoost `0.5003`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
