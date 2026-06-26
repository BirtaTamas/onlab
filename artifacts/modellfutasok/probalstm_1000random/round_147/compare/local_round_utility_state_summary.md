# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv`
- round_num: `11`
- rows: `216`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 216 | 1.000 | 0.709459 | 0.743757 | -0.034298 | 159 | 57 | 0.092593 | 0.092593 |
| active/recent utility | 216 | 1.000 | 0.709459 | 0.743757 | -0.034298 | 159 | 57 | 0.092593 | 0.092593 |
| strong utility action | 191 | 0.884 | 0.705194 | 0.741291 | -0.036098 | 141 | 50 | 0.094241 | 0.094241 |
| utility damage | 20 | 0.093 | 0.802483 | 0.768183 | 0.034300 | 1 | 19 | 0.000000 | 0.000000 |
| active smoke/inferno | 174 | 0.806 | 0.707502 | 0.739427 | -0.031925 | 126 | 48 | 0.103448 | 0.103448 |
| recent utility last 5s | 32 | 0.148 | 0.716196 | 0.786402 | -0.070206 | 31 | 1 | 0.000000 | 0.000000 |
| flash effect present | 216 | 1.000 | 0.709459 | 0.743757 | -0.034298 | 159 | 57 | 0.092593 | 0.092593 |

## Active Smoke/Inferno Intervals

- `8.5s` - `36.0s`, rows `56`
- `45.0s` - `81.5s`, rows `74`
- `85.0s` - `106.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `99.5`, LSTM `0.2396`, XGBoost `0.4401`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.2108`, XGBoost `0.3629`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.2116`, XGBoost `0.3629`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.6216`, XGBoost `0.7567`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `2`
- seconds `100.5`, LSTM `0.3122`, XGBoost `0.4427`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.3112`, XGBoost `0.4401`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.2349`, XGBoost `0.3634`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.8027`, XGBoost `0.6765`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.8012`, XGBoost `0.6765`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.2412`, XGBoost `0.3634`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
