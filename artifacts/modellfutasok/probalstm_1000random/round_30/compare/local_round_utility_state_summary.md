# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `4`
- rows: `267`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 267 | 1.000 | 0.393654 | 0.487976 | -0.094321 | 45 | 222 | 0.217228 | 0.217228 |
| active/recent utility | 267 | 1.000 | 0.393654 | 0.487976 | -0.094321 | 45 | 222 | 0.217228 | 0.217228 |
| strong utility action | 217 | 0.813 | 0.322840 | 0.423972 | -0.101132 | 45 | 172 | 0.110599 | 0.110599 |
| utility damage | 20 | 0.075 | 0.351209 | 0.509137 | -0.157927 | 0 | 20 | 0.350000 | 0.350000 |
| active smoke/inferno | 197 | 0.738 | 0.305971 | 0.408135 | -0.102165 | 39 | 158 | 0.086294 | 0.086294 |
| recent utility last 5s | 10 | 0.037 | 0.428765 | 0.424502 | 0.004264 | 6 | 4 | 0.000000 | 0.000000 |
| flash effect present | 267 | 1.000 | 0.393654 | 0.487976 | -0.094321 | 45 | 222 | 0.217228 | 0.217228 |

## Active Smoke/Inferno Intervals

- `9.5s` - `107.5s`, rows `197`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `45.0`, LSTM `0.1749`, XGBoost `0.4274`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.1787`, XGBoost `0.4295`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.1824`, XGBoost `0.4274`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.1802`, XGBoost `0.4209`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.1938`, XGBoost `0.4291`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.1893`, XGBoost `0.4245`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.1934`, XGBoost `0.4274`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.1938`, XGBoost `0.4274`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `113.5`, LSTM `0.6142`, XGBoost `0.8476`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `53.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.1882`, XGBoost `0.4209`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
