# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `7`
- rows: `244`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 244 | 1.000 | 0.165286 | 0.191689 | -0.026404 | 219 | 25 | 0.938525 | 1.000000 |
| active/recent utility | 244 | 1.000 | 0.165286 | 0.191689 | -0.026404 | 219 | 25 | 0.938525 | 1.000000 |
| strong utility action | 147 | 0.602 | 0.219395 | 0.259865 | -0.040469 | 127 | 20 | 0.918367 | 1.000000 |
| utility damage | 10 | 0.041 | 0.135503 | 0.250127 | -0.114623 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 147 | 0.602 | 0.219395 | 0.259865 | -0.040469 | 127 | 20 | 0.918367 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 244 | 1.000 | 0.165286 | 0.191689 | -0.026404 | 219 | 25 | 0.938525 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `81.5s`, rows `147`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.0`, LSTM `0.0859`, XGBoost `0.2468`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.0850`, XGBoost `0.2456`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.1017`, XGBoost `0.2446`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.1109`, XGBoost `0.2501`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.1074`, XGBoost `0.2456`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.1132`, XGBoost `0.2501`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.1147`, XGBoost `0.2501`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.1131`, XGBoost `0.2482`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.1224`, XGBoost `0.2501`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.1172`, XGBoost `0.2446`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
