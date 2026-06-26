# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `9`
- rows: `251`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 251 | 1.000 | 0.415409 | 0.477699 | -0.062290 | 240 | 11 | 0.470120 | 0.442231 |
| active/recent utility | 251 | 1.000 | 0.415409 | 0.477699 | -0.062290 | 240 | 11 | 0.470120 | 0.442231 |
| strong utility action | 211 | 0.841 | 0.462329 | 0.533623 | -0.071293 | 207 | 4 | 0.407583 | 0.374408 |
| utility damage | 10 | 0.040 | 0.115993 | 0.393453 | -0.277460 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 205 | 0.817 | 0.453648 | 0.526262 | -0.072614 | 201 | 4 | 0.419512 | 0.385366 |
| recent utility last 5s | 10 | 0.040 | 0.767047 | 0.786642 | -0.019594 | 10 | 0 | 0.000000 | 0.000000 |
| flash effect present | 251 | 1.000 | 0.415409 | 0.477699 | -0.062290 | 240 | 11 | 0.470120 | 0.442231 |

## Active Smoke/Inferno Intervals

- `6.5s` - `46.0s`, rows `80`
- `47.0s` - `109.0s`, rows `125`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `77.0`, LSTM `0.1131`, XGBoost `0.4647`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `33.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.1237`, XGBoost `0.4678`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `33.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.1331`, XGBoost `0.4647`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `33.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.1339`, XGBoost `0.4647`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `33.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.1396`, XGBoost `0.4647`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `33.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.1416`, XGBoost `0.4647`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `33.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.1212`, XGBoost `0.4385`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.1597`, XGBoost `0.4678`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `33.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.1650`, XGBoost `0.4656`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `33.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.2226`, XGBoost `0.4360`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
