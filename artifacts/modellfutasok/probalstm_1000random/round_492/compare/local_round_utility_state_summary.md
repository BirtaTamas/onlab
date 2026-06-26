# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `17`
- rows: `171`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 171 | 1.000 | 0.425358 | 0.449710 | -0.024352 | 129 | 42 | 0.479532 | 0.438596 |
| active/recent utility | 171 | 1.000 | 0.425358 | 0.449710 | -0.024352 | 129 | 42 | 0.479532 | 0.438596 |
| strong utility action | 168 | 0.982 | 0.423765 | 0.448956 | -0.025191 | 128 | 40 | 0.482143 | 0.428571 |
| utility damage | 10 | 0.058 | 0.548713 | 0.545894 | 0.002819 | 4 | 6 | 0.000000 | 0.000000 |
| active smoke/inferno | 158 | 0.924 | 0.418494 | 0.446297 | -0.027804 | 126 | 32 | 0.481013 | 0.392405 |
| recent utility last 5s | 20 | 0.117 | 0.506467 | 0.519213 | -0.012746 | 11 | 9 | 0.500000 | 0.500000 |
| flash effect present | 171 | 1.000 | 0.425358 | 0.449710 | -0.024352 | 129 | 42 | 0.479532 | 0.438596 |

## Active Smoke/Inferno Intervals

- `6.5s` - `85.0s`, rows `158`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.5`, LSTM `0.1893`, XGBoost `0.3557`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.2671`, XGBoost `0.1035`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `6.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.2050`, XGBoost `0.3662`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.2064`, XGBoost `0.3557`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.2910`, XGBoost `0.1430`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.2249`, XGBoost `0.3630`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.2272`, XGBoost `0.3618`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.2271`, XGBoost `0.3588`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.2289`, XGBoost `0.3557`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.2371`, XGBoost `0.3618`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
