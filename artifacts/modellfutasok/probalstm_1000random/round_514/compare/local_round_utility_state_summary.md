# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m2-mirage.csv`
- round_num: `5`
- rows: `278`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 278 | 1.000 | 0.404532 | 0.380143 | 0.024389 | 97 | 181 | 0.496403 | 0.989209 |
| active/recent utility | 278 | 1.000 | 0.404532 | 0.380143 | 0.024389 | 97 | 181 | 0.496403 | 0.989209 |
| strong utility action | 220 | 0.791 | 0.449590 | 0.422494 | 0.027096 | 63 | 157 | 0.454545 | 0.986364 |
| utility damage | 31 | 0.112 | 0.302610 | 0.314059 | -0.011449 | 18 | 13 | 0.967742 | 1.000000 |
| active smoke/inferno | 207 | 0.745 | 0.456609 | 0.427392 | 0.029217 | 56 | 151 | 0.420290 | 0.985507 |
| recent utility last 5s | 10 | 0.036 | 0.477015 | 0.469326 | 0.007689 | 3 | 7 | 1.000000 | 1.000000 |
| flash effect present | 278 | 1.000 | 0.404532 | 0.380143 | 0.024389 | 97 | 181 | 0.496403 | 0.989209 |

## Active Smoke/Inferno Intervals

- `6.5s` - `59.0s`, rows `106`
- `69.5s` - `119.5s`, rows `101`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `103.5`, LSTM `0.7585`, XGBoost `0.6129`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.1842`, XGBoost `0.3205`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `108.5`, LSTM `0.1672`, XGBoost `0.3011`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.6004`, XGBoost `0.4835`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.3153`, XGBoost `0.4320`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `14.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.5169`, XGBoost `0.4015`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `14.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.5456`, XGBoost `0.4308`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.1872`, XGBoost `0.3011`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.5969`, XGBoost `0.4835`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.5830`, XGBoost `0.4705`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
