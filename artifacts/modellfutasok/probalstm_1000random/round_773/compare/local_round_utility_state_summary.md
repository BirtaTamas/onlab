# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `14`
- rows: `185`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 185 | 1.000 | 0.748201 | 0.817013 | -0.068812 | 23 | 162 | 0.891892 | 0.897297 |
| active/recent utility | 185 | 1.000 | 0.748201 | 0.817013 | -0.068812 | 23 | 162 | 0.891892 | 0.897297 |
| strong utility action | 137 | 0.741 | 0.745354 | 0.823925 | -0.078571 | 13 | 124 | 0.854015 | 0.861314 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 137 | 0.741 | 0.745354 | 0.823925 | -0.078571 | 13 | 124 | 0.854015 | 0.861314 |
| recent utility last 5s | 10 | 0.054 | 0.887360 | 0.924104 | -0.036743 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 185 | 1.000 | 0.748201 | 0.817013 | -0.068812 | 23 | 162 | 0.891892 | 0.897297 |

## Active Smoke/Inferno Intervals

- `9.5s` - `50.5s`, rows `83`
- `54.5s` - `81.0s`, rows `54`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.0`, LSTM `0.1207`, XGBoost `0.3900`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `90.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.1292`, XGBoost `0.3932`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `90.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1406`, XGBoost `0.3980`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `90.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1375`, XGBoost `0.3806`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `90.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.7161`, XGBoost `0.9515`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.1641`, XGBoost `0.3980`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `90.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.7242`, XGBoost `0.9515`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.7306`, XGBoost `0.9513`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.7366`, XGBoost `0.9515`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.7480`, XGBoost `0.9513`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
