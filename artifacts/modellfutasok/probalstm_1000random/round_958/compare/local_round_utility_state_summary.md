# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `12`
- rows: `135`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.188089 | 0.198724 | -0.010635 | 96 | 39 | 0.940741 | 0.962963 |
| active/recent utility | 135 | 1.000 | 0.188089 | 0.198724 | -0.010635 | 96 | 39 | 0.940741 | 0.962963 |
| strong utility action | 111 | 0.822 | 0.140484 | 0.155407 | -0.014923 | 85 | 26 | 0.945946 | 0.954955 |
| utility damage | 30 | 0.222 | 0.313781 | 0.319780 | -0.005999 | 12 | 18 | 0.800000 | 0.833333 |
| active smoke/inferno | 111 | 0.822 | 0.140484 | 0.155407 | -0.014923 | 85 | 26 | 0.945946 | 0.954955 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 135 | 1.000 | 0.188089 | 0.198724 | -0.010635 | 96 | 39 | 0.940741 | 0.962963 |

## Active Smoke/Inferno Intervals

- `10.5s` - `65.5s`, rows `111`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `20.5`, LSTM `0.0898`, XGBoost `0.2800`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `100.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.1293`, XGBoost `0.3119`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `164.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.1283`, XGBoost `0.2969`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `100.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.1539`, XGBoost `0.3119`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `164.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.1335`, XGBoost `0.2809`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `100.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.1402`, XGBoost `0.2741`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `100.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.1451`, XGBoost `0.2755`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `100.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.1818`, XGBoost `0.3006`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `52.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.4385`, XGBoost `0.3290`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `206.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.2047`, XGBoost `0.3119`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `164.0`, recent_utility `0`
