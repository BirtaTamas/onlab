# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `7`
- rows: `197`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 197 | 1.000 | 0.731035 | 0.761305 | -0.030270 | 100 | 97 | 0.081218 | 0.086294 |
| active/recent utility | 197 | 1.000 | 0.731035 | 0.761305 | -0.030270 | 100 | 97 | 0.081218 | 0.086294 |
| strong utility action | 129 | 0.655 | 0.780243 | 0.787123 | -0.006879 | 59 | 70 | 0.000000 | 0.007752 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 118 | 0.599 | 0.781262 | 0.788341 | -0.007079 | 50 | 68 | 0.000000 | 0.008475 |
| recent utility last 5s | 11 | 0.056 | 0.769319 | 0.774058 | -0.004739 | 9 | 2 | 0.000000 | 0.000000 |
| flash effect present | 197 | 1.000 | 0.731035 | 0.761305 | -0.030270 | 100 | 97 | 0.081218 | 0.086294 |

## Active Smoke/Inferno Intervals

- `7.5s` - `36.5s`, rows `59`
- `55.0s` - `84.0s`, rows `59`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `75.5`, LSTM `0.8397`, XGBoost `0.9711`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.8526`, XGBoost `0.9712`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.8530`, XGBoost `0.9711`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.5417`, XGBoost `0.6363`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.6803`, XGBoost `0.7554`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.6894`, XGBoost `0.7620`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.9253`, XGBoost `0.8536`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.8933`, XGBoost `0.9636`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.5373`, XGBoost `0.6075`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.8932`, XGBoost `0.9632`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
