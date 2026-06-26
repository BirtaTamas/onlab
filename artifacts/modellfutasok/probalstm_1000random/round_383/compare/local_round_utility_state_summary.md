# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-g2-vs-falcons-bo3-VnJ8NRf6cDNnH9OuqiscGr/g2-vs-falcons-m1-ancient.csv`
- round_num: `6`
- rows: `103`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 103 | 1.000 | 0.945008 | 0.981873 | -0.036865 | 0 | 103 | 1.000000 | 1.000000 |
| active/recent utility | 103 | 1.000 | 0.945008 | 0.981873 | -0.036865 | 0 | 103 | 1.000000 | 1.000000 |
| strong utility action | 46 | 0.447 | 0.942791 | 0.978693 | -0.035902 | 0 | 46 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 46 | 0.447 | 0.942791 | 0.978693 | -0.035902 | 0 | 46 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 103 | 1.000 | 0.945008 | 0.981873 | -0.036865 | 0 | 103 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `29.0s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.5`, LSTM `0.9206`, XGBoost `0.9782`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.9221`, XGBoost `0.9782`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.9222`, XGBoost `0.9782`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.9269`, XGBoost `0.9782`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.9305`, XGBoost `0.9786`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.9332`, XGBoost `0.9782`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.9349`, XGBoost `0.9786`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.9381`, XGBoost `0.9795`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.9391`, XGBoost `0.9794`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.9393`, XGBoost `0.9794`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
