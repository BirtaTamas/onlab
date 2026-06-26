# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `10`
- rows: `213`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 213 | 1.000 | 0.246259 | 0.347723 | -0.101463 | 210 | 3 | 0.962441 | 0.953052 |
| active/recent utility | 213 | 1.000 | 0.246259 | 0.347723 | -0.101463 | 210 | 3 | 0.962441 | 0.953052 |
| strong utility action | 174 | 0.817 | 0.240794 | 0.338139 | -0.097345 | 172 | 2 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.094 | 0.233284 | 0.345412 | -0.112127 | 20 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 158 | 0.742 | 0.248087 | 0.336963 | -0.088876 | 156 | 2 | 1.000000 | 1.000000 |
| recent utility last 5s | 28 | 0.131 | 0.216195 | 0.350650 | -0.134455 | 28 | 0 | 1.000000 | 1.000000 |
| flash effect present | 213 | 1.000 | 0.246259 | 0.347723 | -0.101463 | 210 | 3 | 0.962441 | 0.953052 |

## Active Smoke/Inferno Intervals

- `9.0s` - `41.5s`, rows `66`
- `43.0s` - `81.5s`, rows `78`
- `85.5s` - `92.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `7.0`, LSTM `0.1362`, XGBoost `0.3524`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `8.0`, recent_utility `2`
- seconds `78.5`, LSTM `0.1413`, XGBoost `0.3533`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.1478`, XGBoost `0.3524`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `8.0`, recent_utility `2`
- seconds `5.0`, LSTM `0.1548`, XGBoost `0.3559`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `8.0`, recent_utility `3`
- seconds `5.5`, LSTM `0.1526`, XGBoost `0.3532`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `8.0`, recent_utility `3`
- seconds `6.0`, LSTM `0.1552`, XGBoost `0.3524`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `8.0`, recent_utility `2`
- seconds `4.5`, LSTM `0.1628`, XGBoost `0.3552`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `8.0`, recent_utility `2`
- seconds `3.5`, LSTM `0.1662`, XGBoost `0.3572`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `79.0`, LSTM `0.1503`, XGBoost `0.3409`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.1534`, XGBoost `0.3438`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `8.0`, recent_utility `1`
