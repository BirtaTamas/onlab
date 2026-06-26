# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m3-overpass.csv`
- round_num: `11`
- rows: `217`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 217 | 1.000 | 0.850008 | 0.879368 | -0.029359 | 61 | 156 | 1.000000 | 1.000000 |
| active/recent utility | 217 | 1.000 | 0.850008 | 0.879368 | -0.029359 | 61 | 156 | 1.000000 | 1.000000 |
| strong utility action | 179 | 0.825 | 0.849471 | 0.888262 | -0.038791 | 43 | 136 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.092 | 0.700878 | 0.807778 | -0.106899 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 179 | 0.825 | 0.849471 | 0.888262 | -0.038791 | 43 | 136 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 217 | 1.000 | 0.850008 | 0.879368 | -0.029359 | 61 | 156 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `68.5s`, rows `123`
- `70.0s` - `75.0s`, rows `11`
- `78.0s` - `100.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `88.5`, LSTM `0.6763`, XGBoost `0.8755`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `16.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.6864`, XGBoost `0.8844`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.6862`, XGBoost `0.8838`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.6912`, XGBoost `0.8863`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `30.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.6937`, XGBoost `0.8863`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `30.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.7025`, XGBoost `0.8844`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `28.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.7060`, XGBoost `0.8844`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `30.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.7115`, XGBoost `0.8868`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `30.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.7139`, XGBoost `0.8755`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `16.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.7327`, XGBoost `0.8896`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `30.0`, recent_utility `0`
