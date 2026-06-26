# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `20`
- rows: `177`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 177 | 1.000 | 0.299705 | 0.382863 | -0.083158 | 177 | 0 | 0.694915 | 0.435028 |
| active/recent utility | 177 | 1.000 | 0.299705 | 0.382863 | -0.083158 | 177 | 0 | 0.694915 | 0.435028 |
| strong utility action | 160 | 0.904 | 0.307504 | 0.394088 | -0.086583 | 160 | 0 | 0.662500 | 0.406250 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 150 | 0.847 | 0.300313 | 0.386870 | -0.086557 | 150 | 0 | 0.640000 | 0.426667 |
| recent utility last 5s | 10 | 0.056 | 0.415377 | 0.502346 | -0.086970 | 10 | 0 | 1.000000 | 0.100000 |
| flash effect present | 177 | 1.000 | 0.299705 | 0.382863 | -0.083158 | 177 | 0 | 0.694915 | 0.435028 |

## Active Smoke/Inferno Intervals

- `8.0s` - `59.5s`, rows `104`
- `65.5s` - `88.0s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.0`, LSTM `0.5133`, XGBoost `0.6725`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.3736`, XGBoost `0.5301`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.5183`, XGBoost `0.6729`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.3795`, XGBoost `0.5322`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.3734`, XGBoost `0.5259`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5203`, XGBoost `0.6725`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5220`, XGBoost `0.6717`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.3825`, XGBoost `0.5322`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.2671`, XGBoost `0.4161`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.3852`, XGBoost `0.5330`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
