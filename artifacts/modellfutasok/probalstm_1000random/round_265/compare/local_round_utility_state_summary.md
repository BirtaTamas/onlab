# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `3`
- rows: `174`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 174 | 1.000 | 0.943084 | 0.979740 | -0.036656 | 0 | 174 | 1.000000 | 1.000000 |
| active/recent utility | 174 | 1.000 | 0.943084 | 0.979740 | -0.036656 | 0 | 174 | 1.000000 | 1.000000 |
| strong utility action | 152 | 0.874 | 0.944225 | 0.979886 | -0.035661 | 0 | 152 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.115 | 0.934043 | 0.972606 | -0.038563 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 142 | 0.816 | 0.951277 | 0.980822 | -0.029546 | 0 | 142 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.057 | 0.844095 | 0.966591 | -0.122496 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 174 | 1.000 | 0.943084 | 0.979740 | -0.036656 | 0 | 174 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `59.0s`, rows `98`
- `65.0s` - `86.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `4.5`, LSTM `0.8337`, XGBoost `0.9675`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.8384`, XGBoost `0.9684`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.8362`, XGBoost `0.9659`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.8375`, XGBoost `0.9668`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.8416`, XGBoost `0.9668`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.8461`, XGBoost `0.9659`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.0`, LSTM `0.8493`, XGBoost `0.9659`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.8507`, XGBoost `0.9659`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.8523`, XGBoost `0.9668`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.5`, LSTM `0.8552`, XGBoost `0.9659`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
