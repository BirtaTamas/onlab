# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `5`
- rows: `284`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 284 | 1.000 | 0.345912 | 0.275850 | 0.070063 | 101 | 183 | 0.626761 | 0.823944 |
| active/recent utility | 284 | 1.000 | 0.345912 | 0.275850 | 0.070063 | 101 | 183 | 0.626761 | 0.823944 |
| strong utility action | 132 | 0.465 | 0.435225 | 0.344612 | 0.090613 | 28 | 104 | 0.469697 | 0.719697 |
| utility damage | 22 | 0.077 | 0.524459 | 0.453375 | 0.071084 | 0 | 22 | 0.500000 | 0.545455 |
| active smoke/inferno | 132 | 0.465 | 0.435225 | 0.344612 | 0.090613 | 28 | 104 | 0.469697 | 0.719697 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 284 | 1.000 | 0.345912 | 0.275850 | 0.070063 | 101 | 183 | 0.626761 | 0.823944 |

## Active Smoke/Inferno Intervals

- `6.5s` - `58.0s`, rows `104`
- `99.5s` - `106.0s`, rows `14`
- `122.5s` - `129.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.0`, LSTM `0.7214`, XGBoost `0.5579`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5262`, XGBoost `0.3699`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.7141`, XGBoost `0.5579`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5253`, XGBoost `0.3697`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.5240`, XGBoost `0.3688`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.7128`, XGBoost `0.5581`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.7124`, XGBoost `0.5579`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5253`, XGBoost `0.3709`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.5241`, XGBoost `0.3699`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.7118`, XGBoost `0.5579`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
