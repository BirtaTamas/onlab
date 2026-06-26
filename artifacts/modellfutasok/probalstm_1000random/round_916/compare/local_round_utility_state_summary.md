# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `5`
- rows: `230`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.445768 | 0.525315 | -0.079547 | 225 | 5 | 0.721739 | 0.365217 |
| active/recent utility | 230 | 1.000 | 0.445768 | 0.525315 | -0.079547 | 225 | 5 | 0.721739 | 0.365217 |
| strong utility action | 199 | 0.865 | 0.446727 | 0.517295 | -0.070568 | 194 | 5 | 0.768844 | 0.366834 |
| utility damage | 10 | 0.043 | 0.422539 | 0.481992 | -0.059452 | 10 | 0 | 1.000000 | 0.900000 |
| active smoke/inferno | 191 | 0.830 | 0.450678 | 0.519885 | -0.069207 | 186 | 5 | 0.759162 | 0.340314 |
| recent utility last 5s | 10 | 0.043 | 0.352030 | 0.453855 | -0.101825 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.445768 | 0.525315 | -0.079547 | 225 | 5 | 0.721739 | 0.365217 |

## Active Smoke/Inferno Intervals

- `6.5s` - `59.0s`, rows `106`
- `69.5s` - `111.5s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `88.5`, LSTM `0.3015`, XGBoost `0.4913`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.5320`, XGBoost `0.7209`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.5346`, XGBoost `0.7197`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.5271`, XGBoost `0.7118`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.5271`, XGBoost `0.7118`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.5365`, XGBoost `0.7209`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.5388`, XGBoost `0.7217`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.5408`, XGBoost `0.7201`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.5409`, XGBoost `0.7201`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.5344`, XGBoost `0.7118`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
