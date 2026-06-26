# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `8`
- rows: `173`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.125357 | 0.210253 | -0.084896 | 173 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 173 | 1.000 | 0.125357 | 0.210253 | -0.084896 | 173 | 0 | 1.000000 | 1.000000 |
| strong utility action | 138 | 0.798 | 0.152426 | 0.250397 | -0.097971 | 138 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 135 | 0.780 | 0.153440 | 0.249261 | -0.095822 | 135 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.058 | 0.126735 | 0.296264 | -0.169529 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 173 | 1.000 | 0.125357 | 0.210253 | -0.084896 | 173 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `3.0s` - `63.0s`, rows `121`
- `77.5s` - `84.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `2.0`, LSTM `0.1031`, XGBoost `0.3054`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.1134`, XGBoost `0.3054`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.1038`, XGBoost `0.2938`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.1251`, XGBoost `0.3054`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.1304`, XGBoost `0.3034`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `59.0`, LSTM `0.1269`, XGBoost `0.2954`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.1356`, XGBoost `0.2954`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.1307`, XGBoost `0.2874`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.1311`, XGBoost `0.2866`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.5`, LSTM `0.1327`, XGBoost `0.2861`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
