# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `16`
- rows: `191`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 191 | 1.000 | 0.354570 | 0.373614 | -0.019044 | 111 | 80 | 0.837696 | 1.000000 |
| active/recent utility | 191 | 1.000 | 0.354570 | 0.373614 | -0.019044 | 111 | 80 | 0.837696 | 1.000000 |
| strong utility action | 148 | 0.775 | 0.355010 | 0.358948 | -0.003938 | 75 | 73 | 0.790541 | 1.000000 |
| utility damage | 10 | 0.052 | 0.373553 | 0.440929 | -0.067376 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 148 | 0.775 | 0.355010 | 0.358948 | -0.003938 | 75 | 73 | 0.790541 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 191 | 1.000 | 0.354570 | 0.373614 | -0.019044 | 111 | 80 | 0.837696 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `34.5s`, rows `51`
- `47.0s` - `95.0s`, rows `97`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `92.5`, LSTM `0.0375`, XGBoost `0.3333`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.0098`, XGBoost `0.1395`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.0228`, XGBoost `0.1524`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.0277`, XGBoost `0.1540`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.0148`, XGBoost `0.1392`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.3178`, XGBoost `0.4406`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.3195`, XGBoost `0.4412`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.0161`, XGBoost `0.1375`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.3377`, XGBoost `0.2212`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.0339`, XGBoost `0.1467`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
