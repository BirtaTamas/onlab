# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-872ZDvS9tk2PrtGeXVe8dJ/aurora-vs-heroic-m1-train-p3.csv`
- round_num: `2`
- rows: `219`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 219 | 1.000 | 0.481830 | 0.524585 | -0.042755 | 207 | 12 | 0.187215 | 0.155251 |
| active/recent utility | 219 | 1.000 | 0.481830 | 0.524585 | -0.042755 | 207 | 12 | 0.187215 | 0.155251 |
| strong utility action | 189 | 0.863 | 0.510121 | 0.557343 | -0.047222 | 179 | 10 | 0.105820 | 0.105820 |
| utility damage | 39 | 0.178 | 0.533039 | 0.624038 | -0.090999 | 39 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 189 | 0.863 | 0.510121 | 0.557343 | -0.047222 | 179 | 10 | 0.105820 | 0.105820 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 219 | 1.000 | 0.481830 | 0.524585 | -0.042755 | 207 | 12 | 0.187215 | 0.155251 |

## Active Smoke/Inferno Intervals

- `8.0s` - `95.0s`, rows `175`
- `96.5s` - `103.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.0`, LSTM `0.5035`, XGBoost `0.6618`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `6.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.5384`, XGBoost `0.6965`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `38.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.5396`, XGBoost `0.6965`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `44.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.5410`, XGBoost `0.6965`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `38.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.5102`, XGBoost `0.6646`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `6.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.5423`, XGBoost `0.6965`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `38.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.5186`, XGBoost `0.6705`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `6.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.5135`, XGBoost `0.6619`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `6.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.5449`, XGBoost `0.6917`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5414`, XGBoost `0.6872`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
