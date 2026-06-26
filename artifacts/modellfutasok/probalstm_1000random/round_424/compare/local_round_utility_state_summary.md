# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-mouz-vs-falcons-bo3-OIe4ELGS25ekkV8Rf6FbR4/mouz-vs-falcons-m3-mirage.csv`
- round_num: `16`
- rows: `182`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 182 | 1.000 | 0.215145 | 0.227976 | -0.012832 | 141 | 41 | 0.725275 | 0.725275 |
| active/recent utility | 182 | 1.000 | 0.215145 | 0.227976 | -0.012832 | 141 | 41 | 0.725275 | 0.725275 |
| strong utility action | 103 | 0.566 | 0.312048 | 0.336182 | -0.024134 | 71 | 32 | 0.631068 | 0.631068 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 103 | 0.566 | 0.312048 | 0.336182 | -0.024134 | 71 | 32 | 0.631068 | 0.631068 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 182 | 1.000 | 0.215145 | 0.227976 | -0.012832 | 141 | 41 | 0.725275 | 0.725275 |

## Active Smoke/Inferno Intervals

- `6.0s` - `57.0s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `42.0`, LSTM `0.1081`, XGBoost `0.3603`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.1272`, XGBoost `0.3676`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.1100`, XGBoost `0.2999`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.1910`, XGBoost `0.3660`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.1942`, XGBoost `0.3673`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.2425`, XGBoost `0.3668`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.2513`, XGBoost `0.3699`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.2456`, XGBoost `0.3604`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.2537`, XGBoost `0.3604`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0693`, XGBoost `0.1645`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
