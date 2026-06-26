# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `12`
- rows: `168`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 168 | 1.000 | 0.304425 | 0.293682 | 0.010742 | 65 | 103 | 0.607143 | 0.607143 |
| active/recent utility | 168 | 1.000 | 0.304425 | 0.293682 | 0.010742 | 65 | 103 | 0.607143 | 0.607143 |
| strong utility action | 158 | 0.940 | 0.283234 | 0.275547 | 0.007687 | 65 | 93 | 0.645570 | 0.645570 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 148 | 0.881 | 0.259512 | 0.254951 | 0.004561 | 65 | 83 | 0.689189 | 0.689189 |
| recent utility last 5s | 20 | 0.119 | 0.624997 | 0.577059 | 0.047938 | 0 | 20 | 0.000000 | 0.000000 |
| flash effect present | 168 | 1.000 | 0.304425 | 0.293682 | 0.010742 | 65 | 103 | 0.607143 | 0.607143 |

## Active Smoke/Inferno Intervals

- `10.0s` - `83.5s`, rows `148`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.0`, LSTM `0.4686`, XGBoost `0.2991`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.1372`, XGBoost `0.2563`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.1538`, XGBoost `0.2431`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.1631`, XGBoost `0.2431`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6604`, XGBoost `0.5841`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.6073`, XGBoost `0.5315`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6573`, XGBoost `0.5841`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.1834`, XGBoost `0.2563`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.6003`, XGBoost `0.5285`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.1720`, XGBoost `0.2434`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
