# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-faze-bo3-1414ljxN3FRmXv6-03KYFL/spirit-vs-faze-m2-mirage.csv`
- round_num: `1`
- rows: `150`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 150 | 1.000 | 0.345309 | 0.441830 | -0.096521 | 147 | 3 | 0.680000 | 0.453333 |
| active/recent utility | 150 | 1.000 | 0.345309 | 0.441830 | -0.096521 | 147 | 3 | 0.680000 | 0.453333 |
| strong utility action | 50 | 0.333 | 0.307510 | 0.424637 | -0.117127 | 47 | 3 | 0.760000 | 0.640000 |
| utility damage | 10 | 0.067 | 0.418508 | 0.529371 | -0.110863 | 10 | 0 | 0.800000 | 0.800000 |
| active smoke/inferno | 50 | 0.333 | 0.307510 | 0.424637 | -0.117127 | 47 | 3 | 0.760000 | 0.640000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 150 | 1.000 | 0.345309 | 0.441830 | -0.096521 | 147 | 3 | 0.680000 | 0.453333 |

## Active Smoke/Inferno Intervals

- `16.5s` - `41.0s`, rows `50`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.0`, LSTM `0.1523`, XGBoost `0.4157`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.1723`, XGBoost `0.4129`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.3527`, XGBoost `0.5827`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.1958`, XGBoost `0.4129`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.4787`, XGBoost `0.6937`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5366`, XGBoost `0.7486`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.3742`, XGBoost `0.5827`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.0450`, XGBoost `0.2498`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.0489`, XGBoost `0.2498`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.0510`, XGBoost `0.2498`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
