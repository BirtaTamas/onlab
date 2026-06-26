# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-g2-vs-falcons-bo3-VnJ8NRf6cDNnH9OuqiscGr/g2-vs-falcons-m1-ancient.csv`
- round_num: `15`
- rows: `242`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 242 | 1.000 | 0.587478 | 0.682118 | -0.094640 | 21 | 221 | 0.822314 | 0.933884 |
| active/recent utility | 242 | 1.000 | 0.587478 | 0.682118 | -0.094640 | 21 | 221 | 0.822314 | 0.933884 |
| strong utility action | 208 | 0.860 | 0.571529 | 0.674340 | -0.102811 | 12 | 196 | 0.846154 | 0.966346 |
| utility damage | 27 | 0.112 | 0.565027 | 0.603202 | -0.038174 | 3 | 24 | 0.814815 | 0.925926 |
| active smoke/inferno | 198 | 0.818 | 0.575420 | 0.682033 | -0.106612 | 11 | 187 | 0.868687 | 0.964646 |
| recent utility last 5s | 10 | 0.041 | 0.494487 | 0.522032 | -0.027544 | 1 | 9 | 0.400000 | 1.000000 |
| flash effect present | 242 | 1.000 | 0.587478 | 0.682118 | -0.094640 | 21 | 221 | 0.822314 | 0.933884 |

## Active Smoke/Inferno Intervals

- `6.5s` - `52.5s`, rows `93`
- `57.5s` - `87.5s`, rows `61`
- `92.5s` - `114.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `70.5`, LSTM `0.5489`, XGBoost `0.7585`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.5612`, XGBoost `0.7584`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.5575`, XGBoost `0.7538`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.5550`, XGBoost `0.7509`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.5429`, XGBoost `0.7384`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.5568`, XGBoost `0.7509`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.5462`, XGBoost `0.7383`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.5657`, XGBoost `0.7561`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.6064`, XGBoost `0.7961`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5632`, XGBoost `0.7509`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
