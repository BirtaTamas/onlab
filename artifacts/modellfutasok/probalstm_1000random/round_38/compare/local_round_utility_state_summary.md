# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `4`
- rows: `300`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 300 | 1.000 | 0.270412 | 0.362843 | -0.092431 | 266 | 34 | 0.966667 | 0.396667 |
| active/recent utility | 300 | 1.000 | 0.270412 | 0.362843 | -0.092431 | 266 | 34 | 0.966667 | 0.396667 |
| strong utility action | 211 | 0.703 | 0.364228 | 0.492609 | -0.128381 | 211 | 0 | 0.952607 | 0.175355 |
| utility damage | 21 | 0.070 | 0.438575 | 0.583068 | -0.144493 | 21 | 0 | 0.761905 | 0.000000 |
| active smoke/inferno | 204 | 0.680 | 0.360988 | 0.490993 | -0.130005 | 204 | 0 | 0.950980 | 0.181373 |
| recent utility last 5s | 19 | 0.063 | 0.417993 | 0.536958 | -0.118965 | 19 | 0 | 1.000000 | 0.000000 |
| flash effect present | 300 | 1.000 | 0.270412 | 0.362843 | -0.092431 | 266 | 34 | 0.966667 | 0.396667 |

## Active Smoke/Inferno Intervals

- `7.0s` - `101.0s`, rows `189`
- `102.0s` - `109.0s`, rows `15`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `87.0`, LSTM `0.4044`, XGBoost `0.6777`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.4338`, XGBoost `0.6713`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.1015`, XGBoost `0.3334`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.1075`, XGBoost `0.3372`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.1082`, XGBoost `0.3328`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.4622`, XGBoost `0.6794`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.4636`, XGBoost `0.6794`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.1170`, XGBoost `0.3314`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.1247`, XGBoost `0.3361`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.1199`, XGBoost `0.3295`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
