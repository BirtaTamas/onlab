# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-rare-atom-vs-nomads-bo3-2A6RLk5ZJnfAwsBhy_Qbbv/rare-atom-vs-nomads-m1-mirage.csv`
- round_num: `9`
- rows: `116`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 116 | 1.000 | 0.737315 | 0.747003 | -0.009688 | 42 | 74 | 0.991379 | 0.974138 |
| active/recent utility | 116 | 1.000 | 0.737315 | 0.747003 | -0.009688 | 42 | 74 | 0.991379 | 0.974138 |
| strong utility action | 96 | 0.828 | 0.761071 | 0.776190 | -0.015119 | 24 | 72 | 0.989583 | 0.968750 |
| utility damage | 30 | 0.259 | 0.784857 | 0.775223 | 0.009634 | 17 | 13 | 1.000000 | 1.000000 |
| active smoke/inferno | 96 | 0.828 | 0.761071 | 0.776190 | -0.015119 | 24 | 72 | 0.989583 | 0.968750 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 116 | 1.000 | 0.737315 | 0.747003 | -0.009688 | 42 | 74 | 0.991379 | 0.974138 |

## Active Smoke/Inferno Intervals

- `7.5s` - `52.0s`, rows `90`
- `55.0s` - `57.5s`, rows `6`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.0`, LSTM `0.4948`, XGBoost `0.5798`, closer `xgboost`, smoke `2`, inferno `4`, utility_damage `19.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.6661`, XGBoost `0.7441`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.5675`, XGBoost `0.4897`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.6720`, XGBoost `0.7441`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.5615`, XGBoost `0.4899`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.5709`, XGBoost `0.5005`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.5112`, XGBoost `0.5796`, closer `xgboost`, smoke `2`, inferno `4`, utility_damage `10.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.7871`, XGBoost `0.7207`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `40.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.7854`, XGBoost `0.7207`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `40.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.5168`, XGBoost `0.5798`, closer `xgboost`, smoke `2`, inferno `4`, utility_damage `25.0`, recent_utility `0`
