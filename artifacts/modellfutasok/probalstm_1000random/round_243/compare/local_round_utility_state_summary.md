# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-g2-vs-falcons-bo3-VnJ8NRf6cDNnH9OuqiscGr/g2-vs-falcons-m1-ancient.csv`
- round_num: `1`
- rows: `157`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.271922 | 0.339193 | -0.067271 | 157 | 0 | 0.770701 | 0.484076 |
| active/recent utility | 111 | 0.707 | 0.177394 | 0.252603 | -0.075209 | 111 | 0 | 0.855856 | 0.684685 |
| strong utility action | 82 | 0.522 | 0.238532 | 0.334705 | -0.096173 | 82 | 0 | 0.804878 | 0.573171 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 82 | 0.522 | 0.238532 | 0.334705 | -0.096173 | 82 | 0 | 0.804878 | 0.573171 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 53 | 0.338 | 0.027985 | 0.077539 | -0.049554 | 53 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `23.0s` - `63.5s`, rows `82`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.0`, LSTM `0.2472`, XGBoost `0.5319`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.2581`, XGBoost `0.5282`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.2599`, XGBoost `0.5282`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.1719`, XGBoost `0.4356`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.2989`, XGBoost `0.5361`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0491`, XGBoost `0.2520`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.3410`, XGBoost `0.5376`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.0670`, XGBoost `0.2606`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.1090`, XGBoost `0.3026`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.0287`, XGBoost `0.2192`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
