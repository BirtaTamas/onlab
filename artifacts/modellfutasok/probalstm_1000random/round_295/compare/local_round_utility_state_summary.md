# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `10`
- rows: `147`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 147 | 1.000 | 0.759982 | 0.751365 | 0.008617 | 70 | 77 | 0.959184 | 0.877551 |
| active/recent utility | 147 | 1.000 | 0.759982 | 0.751365 | 0.008617 | 70 | 77 | 0.959184 | 0.877551 |
| strong utility action | 125 | 0.850 | 0.804834 | 0.794271 | 0.010562 | 63 | 62 | 1.000000 | 0.856000 |
| utility damage | 25 | 0.170 | 0.881847 | 0.858036 | 0.023811 | 15 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 125 | 0.850 | 0.804834 | 0.794271 | 0.010562 | 63 | 62 | 1.000000 | 0.856000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 147 | 1.000 | 0.759982 | 0.751365 | 0.008617 | 70 | 77 | 0.959184 | 0.877551 |

## Active Smoke/Inferno Intervals

- `11.0s` - `73.0s`, rows `125`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.5`, LSTM `0.7707`, XGBoost `0.6279`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `81.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.7653`, XGBoost `0.6253`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `78.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.7362`, XGBoost `0.6286`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `56.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.7166`, XGBoost `0.6157`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `40.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5676`, XGBoost `0.4772`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `21.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.6931`, XGBoost `0.6090`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5572`, XGBoost `0.4785`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.6916`, XGBoost `0.6132`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `23.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5433`, XGBoost `0.4779`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.5400`, XGBoost `0.4766`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `39.0`, recent_utility `0`
