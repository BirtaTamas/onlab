# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m2-nuke.csv`
- round_num: `10`
- rows: `271`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 271 | 1.000 | 0.345530 | 0.354861 | -0.009332 | 176 | 95 | 0.723247 | 0.789668 |
| active/recent utility | 271 | 1.000 | 0.345530 | 0.354861 | -0.009332 | 176 | 95 | 0.723247 | 0.789668 |
| strong utility action | 192 | 0.708 | 0.432517 | 0.442615 | -0.010098 | 106 | 86 | 0.708333 | 0.802083 |
| utility damage | 18 | 0.066 | 0.542292 | 0.488548 | 0.053745 | 0 | 18 | 0.166667 | 0.166667 |
| active smoke/inferno | 192 | 0.708 | 0.432517 | 0.442615 | -0.010098 | 106 | 86 | 0.708333 | 0.802083 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 271 | 1.000 | 0.345530 | 0.354861 | -0.009332 | 176 | 95 | 0.723247 | 0.789668 |

## Active Smoke/Inferno Intervals

- `9.5s` - `105.0s`, rows `192`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.0`, LSTM `0.5871`, XGBoost `0.7308`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.6293`, XGBoost `0.7606`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.4166`, XGBoost `0.2922`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.6251`, XGBoost `0.7495`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.6239`, XGBoost `0.7480`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.6064`, XGBoost `0.7296`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.6089`, XGBoost `0.7308`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.3699`, XGBoost `0.4913`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.4660`, XGBoost `0.3450`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.3594`, XGBoost `0.4796`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
