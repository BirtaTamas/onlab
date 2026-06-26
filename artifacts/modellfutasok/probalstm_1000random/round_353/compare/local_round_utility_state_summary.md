# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-mouz-vs-falcons-bo3-ET1FlQ7LAGQtcSrRzzPcv6/mouz-vs-falcons-m1-dust2.csv`
- round_num: `7`
- rows: `227`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 227 | 1.000 | 0.164214 | 0.218595 | -0.054381 | 211 | 16 | 0.982379 | 1.000000 |
| active/recent utility | 227 | 1.000 | 0.164214 | 0.218595 | -0.054381 | 211 | 16 | 0.982379 | 1.000000 |
| strong utility action | 122 | 0.537 | 0.227345 | 0.297203 | -0.069858 | 114 | 8 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.044 | 0.222562 | 0.229019 | -0.006458 | 5 | 5 | 1.000000 | 1.000000 |
| active smoke/inferno | 122 | 0.537 | 0.227345 | 0.297203 | -0.069858 | 114 | 8 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 227 | 1.000 | 0.164214 | 0.218595 | -0.054381 | 211 | 16 | 0.982379 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `67.5s`, rows `122`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.0`, LSTM `0.3660`, XGBoost `0.4903`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.1002`, XGBoost `0.2215`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.1040`, XGBoost `0.2211`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.1050`, XGBoost `0.2214`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.1056`, XGBoost `0.2215`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.0990`, XGBoost `0.2148`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.1071`, XGBoost `0.2215`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.1028`, XGBoost `0.2168`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.3769`, XGBoost `0.4903`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.3808`, XGBoost `0.4917`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
