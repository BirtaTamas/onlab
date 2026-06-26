# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m2-nuke.csv`
- round_num: `1`
- rows: `212`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 212 | 1.000 | 0.586433 | 0.642368 | -0.055935 | 27 | 185 | 0.872642 | 0.844340 |
| active/recent utility | 212 | 1.000 | 0.586433 | 0.642368 | -0.055935 | 27 | 185 | 0.872642 | 0.844340 |
| strong utility action | 83 | 0.392 | 0.519783 | 0.584498 | -0.064715 | 0 | 83 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 83 | 0.392 | 0.519783 | 0.584498 | -0.064715 | 0 | 83 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 212 | 1.000 | 0.586433 | 0.642368 | -0.055935 | 27 | 185 | 0.872642 | 0.844340 |

## Active Smoke/Inferno Intervals

- `12.0s` - `53.0s`, rows `83`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.0`, LSTM `0.5049`, XGBoost `0.6000`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.5082`, XGBoost `0.6000`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5185`, XGBoost `0.6091`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.5188`, XGBoost `0.6091`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5191`, XGBoost `0.6091`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5193`, XGBoost `0.6091`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.5115`, XGBoost `0.6000`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.5201`, XGBoost `0.6079`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.5211`, XGBoost `0.6077`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.5145`, XGBoost `0.6009`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
