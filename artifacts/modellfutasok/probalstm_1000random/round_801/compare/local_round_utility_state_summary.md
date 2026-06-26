# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `5`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.602401 | 0.739013 | -0.136611 | 2 | 228 | 0.765217 | 0.773913 |
| active/recent utility | 230 | 1.000 | 0.602401 | 0.739013 | -0.136611 | 2 | 228 | 0.765217 | 0.773913 |
| strong utility action | 155 | 0.674 | 0.574212 | 0.727478 | -0.153267 | 0 | 155 | 0.741935 | 0.754839 |
| utility damage | 14 | 0.061 | 0.764426 | 0.873970 | -0.109544 | 0 | 14 | 1.000000 | 1.000000 |
| active smoke/inferno | 155 | 0.674 | 0.574212 | 0.727478 | -0.153267 | 0 | 155 | 0.741935 | 0.754839 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.602401 | 0.739013 | -0.136611 | 2 | 228 | 0.765217 | 0.773913 |

## Active Smoke/Inferno Intervals

- `7.0s` - `53.5s`, rows `94`
- `64.5s` - `87.5s`, rows `47`
- `93.0s` - `99.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.0`, LSTM `0.0816`, XGBoost `0.4922`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.0946`, XGBoost `0.4922`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.0952`, XGBoost `0.4892`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.1000`, XGBoost `0.4922`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.1019`, XGBoost `0.4922`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.1021`, XGBoost `0.4922`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.1072`, XGBoost `0.4922`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.1175`, XGBoost `0.4956`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.1157`, XGBoost `0.4922`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.1166`, XGBoost `0.4922`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
