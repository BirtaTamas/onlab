# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `3`
- rows: `122`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 122 | 1.000 | 0.323409 | 0.325566 | -0.002157 | 64 | 58 | 0.688525 | 0.959016 |
| active/recent utility | 122 | 1.000 | 0.323409 | 0.325566 | -0.002157 | 64 | 58 | 0.688525 | 0.959016 |
| strong utility action | 118 | 0.967 | 0.316262 | 0.320885 | -0.004623 | 64 | 54 | 0.711864 | 0.957627 |
| utility damage | 30 | 0.246 | 0.351010 | 0.373318 | -0.022308 | 16 | 14 | 0.500000 | 0.833333 |
| active smoke/inferno | 108 | 0.885 | 0.298527 | 0.307591 | -0.009064 | 64 | 44 | 0.731481 | 0.953704 |
| recent utility last 5s | 10 | 0.082 | 0.507799 | 0.464462 | 0.043337 | 0 | 10 | 0.500000 | 1.000000 |
| flash effect present | 122 | 1.000 | 0.323409 | 0.325566 | -0.002157 | 64 | 58 | 0.688525 | 0.959016 |

## Active Smoke/Inferno Intervals

- `7.0s` - `60.5s`, rows `108`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `45.5`, LSTM `0.0885`, XGBoost `0.2515`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `43.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.1145`, XGBoost `0.2532`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `43.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.1283`, XGBoost `0.2631`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.1314`, XGBoost `0.2641`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.1374`, XGBoost `0.2673`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.1431`, XGBoost `0.2631`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.1550`, XGBoost `0.2708`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.1815`, XGBoost `0.2908`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `43.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.1614`, XGBoost `0.2705`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.1736`, XGBoost `0.2708`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `30.0`, recent_utility `0`
