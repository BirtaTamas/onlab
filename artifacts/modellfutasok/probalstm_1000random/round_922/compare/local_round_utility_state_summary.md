# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `3`
- rows: `232`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 232 | 1.000 | 0.670027 | 0.641203 | 0.028824 | 162 | 70 | 0.943966 | 0.943966 |
| active/recent utility | 232 | 1.000 | 0.670027 | 0.641203 | 0.028824 | 162 | 70 | 0.943966 | 0.943966 |
| strong utility action | 197 | 0.849 | 0.632176 | 0.595416 | 0.036760 | 155 | 42 | 0.934010 | 0.934010 |
| utility damage | 21 | 0.091 | 0.482487 | 0.492477 | -0.009990 | 9 | 12 | 0.428571 | 0.428571 |
| active smoke/inferno | 188 | 0.810 | 0.634395 | 0.597562 | 0.036833 | 146 | 42 | 0.930851 | 0.930851 |
| recent utility last 5s | 12 | 0.052 | 0.584794 | 0.552426 | 0.032368 | 12 | 0 | 1.000000 | 1.000000 |
| flash effect present | 232 | 1.000 | 0.670027 | 0.641203 | 0.028824 | 162 | 70 | 0.943966 | 0.943966 |

## Active Smoke/Inferno Intervals

- `6.0s` - `56.0s`, rows `101`
- `58.5s` - `101.5s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `101.0`, LSTM `0.8408`, XGBoost `0.9642`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.6294`, XGBoost `0.5135`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.6287`, XGBoost `0.5143`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.6271`, XGBoost `0.5135`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.6272`, XGBoost `0.5156`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.6229`, XGBoost `0.5132`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.8584`, XGBoost `0.9647`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.6311`, XGBoost `0.5252`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.6849`, XGBoost `0.5799`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.6200`, XGBoost `0.5156`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
