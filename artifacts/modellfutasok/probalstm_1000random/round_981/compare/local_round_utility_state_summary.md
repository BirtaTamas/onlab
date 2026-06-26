# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-b8-bo3--nzkpOWiS4qFgkFOwM8Hun/legacy-vs-b8-m2-ancient.csv`
- round_num: `6`
- rows: `123`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 123 | 1.000 | 0.050364 | 0.125035 | -0.074671 | 123 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 123 | 1.000 | 0.050364 | 0.125035 | -0.074671 | 123 | 0 | 1.000000 | 1.000000 |
| strong utility action | 108 | 0.878 | 0.054717 | 0.130593 | -0.075877 | 108 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 98 | 0.797 | 0.057616 | 0.123470 | -0.065855 | 98 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.081 | 0.026308 | 0.200401 | -0.174093 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 123 | 1.000 | 0.050364 | 0.125035 | -0.074671 | 123 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `44.0s`, rows `73`
- `48.5s` - `55.0s`, rows `14`
- `56.0s` - `61.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `4.0`, LSTM `0.0231`, XGBoost `0.2051`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.5`, LSTM `0.0230`, XGBoost `0.2046`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.0242`, XGBoost `0.2046`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.0261`, XGBoost `0.2057`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.0249`, XGBoost `0.2026`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.0`, LSTM `0.0272`, XGBoost `0.2026`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.0259`, XGBoost `0.2001`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.5`, LSTM `0.0273`, XGBoost `0.1955`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.0294`, XGBoost `0.1916`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.5`, LSTM `0.0359`, XGBoost `0.1959`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
