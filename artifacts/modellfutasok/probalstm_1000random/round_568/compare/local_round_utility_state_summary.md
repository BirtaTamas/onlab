# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m1-inferno.csv`
- round_num: `8`
- rows: `270`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 270 | 1.000 | 0.069627 | 0.064643 | 0.004984 | 186 | 84 | 1.000000 | 1.000000 |
| active/recent utility | 270 | 1.000 | 0.069627 | 0.064643 | 0.004984 | 186 | 84 | 1.000000 | 1.000000 |
| strong utility action | 127 | 0.470 | 0.085675 | 0.075166 | 0.010509 | 76 | 51 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 127 | 0.470 | 0.085675 | 0.075166 | 0.010509 | 76 | 51 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.037 | 0.066807 | 0.091716 | -0.024910 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 270 | 1.000 | 0.069627 | 0.064643 | 0.004984 | 186 | 84 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `49.0s`, rows `80`
- `72.5s` - `95.5s`, rows `47`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `75.5`, LSTM `0.3834`, XGBoost `0.2131`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.3800`, XGBoost `0.2141`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.3812`, XGBoost `0.2161`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.3728`, XGBoost `0.2143`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.3702`, XGBoost `0.2179`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.3343`, XGBoost `0.2156`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.3451`, XGBoost `0.2301`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.1643`, XGBoost `0.0951`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.1614`, XGBoost `0.0948`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.1611`, XGBoost `0.0951`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
