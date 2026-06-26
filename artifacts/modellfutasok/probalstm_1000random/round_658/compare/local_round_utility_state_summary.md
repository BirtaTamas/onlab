# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `18`
- rows: `168`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 168 | 1.000 | 0.041999 | 0.131345 | -0.089346 | 168 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 168 | 1.000 | 0.041999 | 0.131345 | -0.089346 | 168 | 0 | 1.000000 | 1.000000 |
| strong utility action | 151 | 0.899 | 0.036613 | 0.115853 | -0.079240 | 151 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 151 | 0.899 | 0.036613 | 0.115853 | -0.079240 | 151 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 168 | 1.000 | 0.041999 | 0.131345 | -0.089346 | 168 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `29.0s`, rows `44`
- `30.5s` - `83.5s`, rows `107`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.5`, LSTM `0.0442`, XGBoost `0.2906`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.0452`, XGBoost `0.2880`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0499`, XGBoost `0.2902`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.0412`, XGBoost `0.2751`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.0245`, XGBoost `0.2520`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0263`, XGBoost `0.2528`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0426`, XGBoost `0.2683`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.0418`, XGBoost `0.2673`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.0503`, XGBoost `0.2700`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0414`, XGBoost `0.2574`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
