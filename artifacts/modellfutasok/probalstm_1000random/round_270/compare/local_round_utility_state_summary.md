# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `11`
- rows: `190`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 190 | 1.000 | 0.786319 | 0.786619 | -0.000300 | 93 | 97 | 0.994737 | 1.000000 |
| active/recent utility | 190 | 1.000 | 0.786319 | 0.786619 | -0.000300 | 93 | 97 | 0.994737 | 1.000000 |
| strong utility action | 156 | 0.821 | 0.791773 | 0.783656 | 0.008118 | 81 | 75 | 1.000000 | 1.000000 |
| utility damage | 23 | 0.121 | 0.763524 | 0.733579 | 0.029945 | 13 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 152 | 0.800 | 0.794889 | 0.787532 | 0.007358 | 77 | 75 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.053 | 0.673369 | 0.639121 | 0.034248 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 190 | 1.000 | 0.786319 | 0.786619 | -0.000300 | 93 | 97 | 0.994737 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `35.0s`, rows `58`
- `37.0s` - `83.5s`, rows `94`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `59.5`, LSTM `0.7002`, XGBoost `0.8437`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.7083`, XGBoost `0.8437`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.7829`, XGBoost `0.6554`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.7811`, XGBoost `0.6556`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.7905`, XGBoost `0.6652`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.7793`, XGBoost `0.6554`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.6942`, XGBoost `0.8178`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.7791`, XGBoost `0.6556`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.7862`, XGBoost `0.6641`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.7110`, XGBoost `0.8329`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
