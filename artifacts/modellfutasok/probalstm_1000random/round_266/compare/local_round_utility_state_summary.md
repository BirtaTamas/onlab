# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `15`
- rows: `266`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 266 | 1.000 | 0.019914 | 0.055051 | -0.035138 | 266 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 266 | 1.000 | 0.019914 | 0.055051 | -0.035138 | 266 | 0 | 1.000000 | 1.000000 |
| strong utility action | 174 | 0.654 | 0.025326 | 0.072663 | -0.047337 | 174 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 164 | 0.617 | 0.020788 | 0.062023 | -0.041235 | 164 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 13 | 0.049 | 0.092545 | 0.246115 | -0.153570 | 13 | 0 | 1.000000 | 1.000000 |
| flash effect present | 266 | 1.000 | 0.019914 | 0.055051 | -0.035138 | 266 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `29.0s`, rows `46`
- `37.0s` - `43.5s`, rows `14`
- `45.5s` - `97.0s`, rows `104`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `6.5`, LSTM `0.0627`, XGBoost `0.2417`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.0685`, XGBoost `0.2428`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.0699`, XGBoost `0.2435`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `6.0`, LSTM `0.0714`, XGBoost `0.2435`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `8.5`, LSTM `0.0747`, XGBoost `0.2439`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.0744`, XGBoost `0.2434`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.0`, LSTM `0.0763`, XGBoost `0.2434`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.0791`, XGBoost `0.2435`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `9.0`, LSTM `0.0865`, XGBoost `0.2501`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1051`, XGBoost `0.2516`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
