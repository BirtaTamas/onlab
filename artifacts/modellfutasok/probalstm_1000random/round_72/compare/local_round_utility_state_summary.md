# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-vitality-vs-faze-bo3-hDX5yjYYbla4cw8aPwAYi3/vitality-vs-faze-m1-nuke.csv`
- round_num: `2`
- rows: `206`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 206 | 1.000 | 0.201889 | 0.204241 | -0.002352 | 104 | 102 | 1.000000 | 1.000000 |
| active/recent utility | 206 | 1.000 | 0.201889 | 0.204241 | -0.002352 | 104 | 102 | 1.000000 | 1.000000 |
| strong utility action | 153 | 0.743 | 0.237806 | 0.243544 | -0.005739 | 66 | 87 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.049 | 0.047914 | 0.089800 | -0.041886 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 143 | 0.694 | 0.225072 | 0.238214 | -0.013143 | 66 | 77 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.049 | 0.419907 | 0.319764 | 0.100143 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 206 | 1.000 | 0.201889 | 0.204241 | -0.002352 | 104 | 102 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `36.5s`, rows `56`
- `38.0s` - `81.0s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `42.0`, LSTM `0.1130`, XGBoost `0.3041`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.1202`, XGBoost `0.3041`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.1292`, XGBoost `0.3041`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0826`, XGBoost `0.2570`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0880`, XGBoost `0.2565`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.0923`, XGBoost `0.2557`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.1040`, XGBoost `0.2570`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.1085`, XGBoost `0.2560`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.1087`, XGBoost `0.2552`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.1594`, XGBoost `0.3041`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
