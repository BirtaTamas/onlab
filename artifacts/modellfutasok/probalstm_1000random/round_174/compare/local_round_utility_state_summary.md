# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `11`
- rows: `212`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 212 | 1.000 | 0.565337 | 0.656744 | -0.091408 | 29 | 183 | 0.674528 | 0.806604 |
| active/recent utility | 212 | 1.000 | 0.565337 | 0.656744 | -0.091408 | 29 | 183 | 0.674528 | 0.806604 |
| strong utility action | 159 | 0.750 | 0.535437 | 0.621338 | -0.085901 | 18 | 141 | 0.647799 | 0.823899 |
| utility damage | 22 | 0.104 | 0.662131 | 0.730992 | -0.068860 | 0 | 22 | 1.000000 | 1.000000 |
| active smoke/inferno | 159 | 0.750 | 0.535437 | 0.621338 | -0.085901 | 18 | 141 | 0.647799 | 0.823899 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 212 | 1.000 | 0.565337 | 0.656744 | -0.091408 | 29 | 183 | 0.674528 | 0.806604 |

## Active Smoke/Inferno Intervals

- `7.5s` - `12.5s`, rows `11`
- `13.5s` - `87.0s`, rows `148`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `86.5`, LSTM `0.1179`, XGBoost `0.4949`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.1227`, XGBoost `0.4949`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.1309`, XGBoost `0.4949`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.1406`, XGBoost `0.4949`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.1706`, XGBoost `0.5152`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.1919`, XGBoost `0.5146`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.2251`, XGBoost `0.5119`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.3210`, XGBoost `0.5954`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.3623`, XGBoost `0.6102`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.4204`, XGBoost `0.6660`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
