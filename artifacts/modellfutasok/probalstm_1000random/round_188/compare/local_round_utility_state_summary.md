# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-vitality-vs-falcons-bo3-8ZTMZQ0BkOa0azICXTbCYv/vitality-vs-falcons-m1-inferno-p4.csv`
- round_num: `5`
- rows: `173`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.152730 | 0.189279 | -0.036548 | 149 | 24 | 0.901734 | 1.000000 |
| active/recent utility | 173 | 1.000 | 0.152730 | 0.189279 | -0.036548 | 149 | 24 | 0.901734 | 1.000000 |
| strong utility action | 152 | 0.879 | 0.113045 | 0.150830 | -0.037785 | 131 | 21 | 0.888158 | 1.000000 |
| utility damage | 10 | 0.058 | 0.441484 | 0.435589 | 0.005894 | 3 | 7 | 0.400000 | 1.000000 |
| active smoke/inferno | 152 | 0.879 | 0.113045 | 0.150830 | -0.037785 | 131 | 21 | 0.888158 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 173 | 1.000 | 0.152730 | 0.189279 | -0.036548 | 149 | 24 | 0.901734 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `86.0s`, rows `152`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.5`, LSTM `0.0895`, XGBoost `0.2665`, closer `lstm`, smoke `1`, inferno `5`, utility_damage `33.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.0914`, XGBoost `0.2671`, closer `lstm`, smoke `1`, inferno `5`, utility_damage `33.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.1353`, XGBoost `0.2999`, closer `lstm`, smoke `1`, inferno `5`, utility_damage `80.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.1305`, XGBoost `0.2665`, closer `lstm`, smoke `1`, inferno `5`, utility_damage `33.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1322`, XGBoost `0.2665`, closer `lstm`, smoke `1`, inferno `5`, utility_damage `33.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1410`, XGBoost `0.2665`, closer `lstm`, smoke `1`, inferno `5`, utility_damage `33.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.0556`, XGBoost `0.1625`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0162`, XGBoost `0.1232`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.1816`, XGBoost `0.2842`, closer `lstm`, smoke `1`, inferno `5`, utility_damage `55.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0545`, XGBoost `0.1540`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
