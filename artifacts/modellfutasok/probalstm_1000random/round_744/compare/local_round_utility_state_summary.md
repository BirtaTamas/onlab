# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `16`
- rows: `153`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 153 | 1.000 | 0.081640 | 0.243198 | -0.161558 | 149 | 4 | 0.967320 | 0.967320 |
| active/recent utility | 153 | 1.000 | 0.081640 | 0.243198 | -0.161558 | 149 | 4 | 0.967320 | 0.967320 |
| strong utility action | 106 | 0.693 | 0.095041 | 0.250350 | -0.155309 | 102 | 4 | 0.952830 | 0.952830 |
| utility damage | 10 | 0.065 | 0.034533 | 0.211925 | -0.177392 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 98 | 0.641 | 0.100504 | 0.258861 | -0.158357 | 94 | 4 | 0.948980 | 0.948980 |
| recent utility last 5s | 10 | 0.065 | 0.028125 | 0.147028 | -0.118902 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 153 | 1.000 | 0.081640 | 0.243198 | -0.161558 | 149 | 4 | 0.967320 | 0.967320 |

## Active Smoke/Inferno Intervals

- `5.0s` - `53.5s`, rows `98`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `42.0`, LSTM `0.0703`, XGBoost `0.3054`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.0721`, XGBoost `0.3054`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.0755`, XGBoost `0.3054`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.0756`, XGBoost `0.3054`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.0767`, XGBoost `0.3054`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0793`, XGBoost `0.3048`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.0238`, XGBoost `0.2472`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.0249`, XGBoost `0.2481`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.0827`, XGBoost `0.3041`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.0216`, XGBoost `0.2407`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
