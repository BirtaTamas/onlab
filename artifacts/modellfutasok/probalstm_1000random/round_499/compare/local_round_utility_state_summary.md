# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-falcons-vs-3dmax-bo3-XHM3Ovc8L9TfLFTYQFrGdT/falcons-vs-3dmax-m3-dust2.csv`
- round_num: `5`
- rows: `277`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 277 | 1.000 | 0.168421 | 0.196336 | -0.027915 | 210 | 67 | 0.989170 | 0.992780 |
| active/recent utility | 277 | 1.000 | 0.168421 | 0.196336 | -0.027915 | 210 | 67 | 0.989170 | 0.992780 |
| strong utility action | 170 | 0.614 | 0.222895 | 0.257308 | -0.034413 | 111 | 59 | 0.988235 | 0.988235 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 170 | 0.614 | 0.222895 | 0.257308 | -0.034413 | 111 | 59 | 0.988235 | 0.988235 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 277 | 1.000 | 0.168421 | 0.196336 | -0.027915 | 210 | 67 | 0.989170 | 0.992780 |

## Active Smoke/Inferno Intervals

- `7.5s` - `60.5s`, rows `107`
- `63.0s` - `69.5s`, rows `14`
- `74.0s` - `98.0s`, rows `49`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `84.0`, LSTM `0.1154`, XGBoost `0.2701`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.1027`, XGBoost `0.2480`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.0997`, XGBoost `0.2446`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.1073`, XGBoost `0.2480`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.1228`, XGBoost `0.2597`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.1154`, XGBoost `0.2512`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.1285`, XGBoost `0.2635`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.1242`, XGBoost `0.2582`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.1524`, XGBoost `0.2836`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.1288`, XGBoost `0.2584`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
