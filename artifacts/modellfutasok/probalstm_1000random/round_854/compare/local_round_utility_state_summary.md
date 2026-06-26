# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m1-inferno.csv`
- round_num: `1`
- rows: `230`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.328469 | 0.331490 | -0.003021 | 103 | 127 | 0.552174 | 0.873913 |
| active/recent utility | 106 | 0.461 | 0.121550 | 0.150156 | -0.028606 | 94 | 12 | 0.962264 | 0.877358 |
| strong utility action | 54 | 0.235 | 0.233237 | 0.274835 | -0.041598 | 42 | 12 | 0.925926 | 0.759259 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 54 | 0.235 | 0.233237 | 0.274835 | -0.041598 | 42 | 12 | 0.925926 | 0.759259 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 100 | 0.435 | 0.103607 | 0.136096 | -0.032488 | 94 | 6 | 1.000000 | 0.870000 |

## Active Smoke/Inferno Intervals

- `62.0s` - `88.5s`, rows `54`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `79.5`, LSTM `0.0646`, XGBoost `0.2398`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `42.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.0624`, XGBoost `0.2359`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `42.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.0526`, XGBoost `0.2246`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `42.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.0412`, XGBoost `0.2112`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `42.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.0811`, XGBoost `0.2061`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `42.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.2913`, XGBoost `0.1825`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.0725`, XGBoost `0.1803`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `42.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.3384`, XGBoost `0.4328`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `42.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.4716`, XGBoost `0.5629`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.4727`, XGBoost `0.5634`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
