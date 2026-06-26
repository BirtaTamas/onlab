# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-the-mongolz-vs-heroic-bo3-lz59_87ZRvJjbdTai7Ev35/heroic-vs-3dmax-m3-ancient.csv`
- round_num: `8`
- rows: `133`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 133 | 1.000 | 0.141016 | 0.238018 | -0.097002 | 132 | 1 | 0.984962 | 0.864662 |
| active/recent utility | 133 | 1.000 | 0.141016 | 0.238018 | -0.097002 | 132 | 1 | 0.984962 | 0.864662 |
| strong utility action | 109 | 0.820 | 0.123596 | 0.231150 | -0.107554 | 108 | 1 | 0.981651 | 0.853211 |
| utility damage | 11 | 0.083 | 0.070971 | 0.262988 | -0.192017 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 109 | 0.820 | 0.123596 | 0.231150 | -0.107554 | 108 | 1 | 0.981651 | 0.853211 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 133 | 1.000 | 0.141016 | 0.238018 | -0.097002 | 132 | 1 | 0.984962 | 0.864662 |

## Active Smoke/Inferno Intervals

- `5.5s` - `37.5s`, rows `65`
- `40.0s` - `61.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.5`, LSTM `0.0983`, XGBoost `0.4852`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.1402`, XGBoost `0.4883`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.0415`, XGBoost `0.2995`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0763`, XGBoost `0.3328`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0540`, XGBoost `0.2995`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0938`, XGBoost `0.3328`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.0373`, XGBoost `0.2703`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0666`, XGBoost `0.2995`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.0328`, XGBoost `0.2632`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.0278`, XGBoost `0.2569`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
