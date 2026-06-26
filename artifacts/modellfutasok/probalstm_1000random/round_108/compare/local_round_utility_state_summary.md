# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `10`
- rows: `160`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 160 | 1.000 | 0.024073 | 0.048523 | -0.024450 | 158 | 2 | 1.000000 | 1.000000 |
| active/recent utility | 160 | 1.000 | 0.024073 | 0.048523 | -0.024450 | 158 | 2 | 1.000000 | 1.000000 |
| strong utility action | 126 | 0.787 | 0.024543 | 0.050521 | -0.025978 | 124 | 2 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 116 | 0.725 | 0.018993 | 0.036088 | -0.017094 | 114 | 2 | 1.000000 | 1.000000 |
| recent utility last 5s | 30 | 0.188 | 0.043613 | 0.111811 | -0.068198 | 30 | 0 | 1.000000 | 1.000000 |
| flash effect present | 160 | 1.000 | 0.024073 | 0.048523 | -0.024450 | 158 | 2 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `5.5s` - `33.0s`, rows `56`
- `50.0s` - `79.5s`, rows `60`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.5`, LSTM `0.0846`, XGBoost `0.2765`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.0`, LSTM `0.0853`, XGBoost `0.2765`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.0870`, XGBoost `0.2765`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.0969`, XGBoost `0.2765`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.1039`, XGBoost `0.2783`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.1239`, XGBoost `0.2765`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.1414`, XGBoost `0.2765`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.5`, LSTM `0.0265`, XGBoost `0.0941`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.0`, LSTM `0.0226`, XGBoost `0.0897`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.5`, LSTM `0.0248`, XGBoost `0.0875`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
