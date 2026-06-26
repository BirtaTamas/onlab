# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `4`
- rows: `106`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 106 | 1.000 | 0.348071 | 0.325744 | 0.022326 | 37 | 69 | 0.726415 | 0.716981 |
| active/recent utility | 106 | 1.000 | 0.348071 | 0.325744 | 0.022326 | 37 | 69 | 0.726415 | 0.716981 |
| strong utility action | 93 | 0.877 | 0.324202 | 0.299372 | 0.024829 | 32 | 61 | 0.827957 | 0.817204 |
| utility damage | 10 | 0.094 | 0.243242 | 0.268296 | -0.025054 | 8 | 2 | 1.000000 | 1.000000 |
| active smoke/inferno | 93 | 0.877 | 0.324202 | 0.299372 | 0.024829 | 32 | 61 | 0.827957 | 0.817204 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 106 | 1.000 | 0.348071 | 0.325744 | 0.022326 | 37 | 69 | 0.726415 | 0.716981 |

## Active Smoke/Inferno Intervals

- `6.5s` - `52.5s`, rows `93`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.5`, LSTM `0.4490`, XGBoost `0.2049`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.3897`, XGBoost `0.2062`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `15.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.3821`, XGBoost `0.2062`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `21.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.3596`, XGBoost `0.2010`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.3411`, XGBoost `0.4989`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.4232`, XGBoost `0.2810`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.4750`, XGBoost `0.3357`, closer `xgboost`, smoke `7`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.4352`, XGBoost `0.2963`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.3076`, XGBoost `0.4454`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.4176`, XGBoost `0.2810`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
