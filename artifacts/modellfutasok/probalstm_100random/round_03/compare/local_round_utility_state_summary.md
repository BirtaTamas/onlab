# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `5`
- rows: `197`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 197 | 1.000 | 0.194228 | 0.192647 | 0.001581 | 136 | 61 | 0.761421 | 0.974619 |
| active/recent utility | 197 | 1.000 | 0.194228 | 0.192647 | 0.001581 | 136 | 61 | 0.761421 | 0.974619 |
| strong utility action | 123 | 0.624 | 0.231720 | 0.224507 | 0.007213 | 81 | 42 | 0.691057 | 0.959350 |
| utility damage | 20 | 0.102 | 0.484463 | 0.477038 | 0.007425 | 8 | 12 | 0.450000 | 1.000000 |
| active smoke/inferno | 123 | 0.624 | 0.231720 | 0.224507 | 0.007213 | 81 | 42 | 0.691057 | 0.959350 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 197 | 1.000 | 0.194228 | 0.192647 | 0.001581 | 136 | 61 | 0.761421 | 0.974619 |

## Active Smoke/Inferno Intervals

- `9.5s` - `48.5s`, rows `79`
- `66.5s` - `88.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.0`, LSTM `0.6579`, XGBoost `0.4642`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.6438`, XGBoost `0.4634`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.6416`, XGBoost `0.4701`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.7866`, XGBoost `0.6247`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.7681`, XGBoost `0.6576`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.3737`, XGBoost `0.4663`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `55.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.5761`, XGBoost `0.4848`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.3753`, XGBoost `0.4638`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `55.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.5658`, XGBoost `0.4848`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.5659`, XGBoost `0.4850`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
