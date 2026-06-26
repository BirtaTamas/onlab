# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `9`
- rows: `252`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 252 | 1.000 | 0.188380 | 0.288605 | -0.100225 | 2 | 250 | 0.134921 | 0.150794 |
| active/recent utility | 252 | 1.000 | 0.188380 | 0.288605 | -0.100225 | 2 | 250 | 0.134921 | 0.150794 |
| strong utility action | 198 | 0.786 | 0.140141 | 0.234151 | -0.094009 | 2 | 196 | 0.070707 | 0.080808 |
| utility damage | 10 | 0.040 | 0.159766 | 0.284512 | -0.124745 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 177 | 0.702 | 0.096356 | 0.192514 | -0.096159 | 2 | 175 | 0.022599 | 0.033898 |
| recent utility last 5s | 32 | 0.127 | 0.340607 | 0.420891 | -0.080284 | 0 | 32 | 0.312500 | 0.312500 |
| flash effect present | 252 | 1.000 | 0.188380 | 0.288605 | -0.100225 | 2 | 250 | 0.134921 | 0.150794 |

## Active Smoke/Inferno Intervals

- `6.0s` - `41.5s`, rows `72`
- `52.0s` - `104.0s`, rows `105`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `89.5`, LSTM `0.0670`, XGBoost `0.2820`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.2279`, XGBoost `0.4196`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.2693`, XGBoost `0.4489`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.4110`, XGBoost `0.5810`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1159`, XGBoost `0.2838`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `115.5`, LSTM `0.6718`, XGBoost `0.8377`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.1361`, XGBoost `0.2964`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `11.5`, LSTM `0.1274`, XGBoost `0.2838`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.1348`, XGBoost `0.2910`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `29.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.1288`, XGBoost `0.2845`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `29.0`, recent_utility `0`
