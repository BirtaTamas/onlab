# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `7`
- rows: `146`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 146 | 1.000 | 0.669307 | 0.702212 | -0.032905 | 7 | 139 | 0.691781 | 1.000000 |
| active/recent utility | 146 | 1.000 | 0.669307 | 0.702212 | -0.032905 | 7 | 139 | 0.691781 | 1.000000 |
| strong utility action | 139 | 0.952 | 0.665031 | 0.697782 | -0.032751 | 7 | 132 | 0.697842 | 1.000000 |
| utility damage | 20 | 0.137 | 0.752533 | 0.786514 | -0.033981 | 0 | 20 | 0.950000 | 1.000000 |
| active smoke/inferno | 129 | 0.884 | 0.681407 | 0.711278 | -0.029871 | 7 | 122 | 0.751938 | 1.000000 |
| recent utility last 5s | 11 | 0.075 | 0.453594 | 0.523941 | -0.070347 | 0 | 11 | 0.000000 | 1.000000 |
| flash effect present | 146 | 1.000 | 0.669307 | 0.702212 | -0.032905 | 7 | 139 | 0.691781 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `70.5s`, rows `129`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.5`, LSTM `0.4137`, XGBoost `0.5756`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `11.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.6942`, XGBoost `0.8177`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `11.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.6995`, XGBoost `0.8181`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `11.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.4702`, XGBoost `0.5853`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.5236`, XGBoost `0.6217`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `11.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.4925`, XGBoost `0.5853`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.4871`, XGBoost `0.5762`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `11.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.4390`, XGBoost `0.5265`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `5.5`, LSTM `0.4452`, XGBoost `0.5237`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `39.0`, LSTM `0.5078`, XGBoost `0.5859`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `11.0`, recent_utility `0`
