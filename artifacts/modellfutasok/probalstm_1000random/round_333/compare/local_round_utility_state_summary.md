# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `16`
- rows: `198`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 198 | 1.000 | 0.270613 | 0.312206 | -0.041594 | 196 | 2 | 0.767677 | 0.484848 |
| active/recent utility | 198 | 1.000 | 0.270613 | 0.312206 | -0.041594 | 196 | 2 | 0.767677 | 0.484848 |
| strong utility action | 145 | 0.732 | 0.279849 | 0.317851 | -0.038002 | 143 | 2 | 0.737931 | 0.489655 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 145 | 0.732 | 0.279849 | 0.317851 | -0.038002 | 143 | 2 | 0.737931 | 0.489655 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 198 | 1.000 | 0.270613 | 0.312206 | -0.041594 | 196 | 2 | 0.767677 | 0.484848 |

## Active Smoke/Inferno Intervals

- `10.0s` - `38.0s`, rows `57`
- `42.5s` - `86.0s`, rows `88`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.0`, LSTM `0.0615`, XGBoost `0.1807`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.0640`, XGBoost `0.1815`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.4606`, XGBoost `0.5528`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.4674`, XGBoost `0.5586`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.4545`, XGBoost `0.5424`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.4653`, XGBoost `0.5528`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4569`, XGBoost `0.5424`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4569`, XGBoost `0.5424`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.4662`, XGBoost `0.5457`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.4686`, XGBoost `0.5460`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
