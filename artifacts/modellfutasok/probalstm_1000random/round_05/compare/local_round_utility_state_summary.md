# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `6`
- rows: `219`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 219 | 1.000 | 0.096338 | 0.159997 | -0.063660 | 11 | 208 | 0.000000 | 0.114155 |
| active/recent utility | 219 | 1.000 | 0.096338 | 0.159997 | -0.063660 | 11 | 208 | 0.000000 | 0.114155 |
| strong utility action | 154 | 0.703 | 0.072439 | 0.109168 | -0.036729 | 11 | 143 | 0.000000 | 0.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 154 | 0.703 | 0.072439 | 0.109168 | -0.036729 | 11 | 143 | 0.000000 | 0.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 219 | 1.000 | 0.096338 | 0.159997 | -0.063660 | 11 | 208 | 0.000000 | 0.114155 |

## Active Smoke/Inferno Intervals

- `7.0s` - `76.5s`, rows `140`
- `84.0s` - `90.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.0`, LSTM `0.1550`, XGBoost `0.3405`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.0669`, XGBoost `0.2055`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.4651`, XGBoost `0.3405`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.4645`, XGBoost `0.3411`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.0822`, XGBoost `0.2055`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.0363`, XGBoost `0.1377`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.0705`, XGBoost `0.1675`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.0713`, XGBoost `0.1675`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.0725`, XGBoost `0.1675`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0429`, XGBoost `0.1313`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
