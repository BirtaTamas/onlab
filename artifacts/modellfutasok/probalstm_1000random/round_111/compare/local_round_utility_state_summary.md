# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `2`
- rows: `151`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 151 | 1.000 | 0.013207 | 0.029366 | -0.016159 | 150 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 151 | 1.000 | 0.013207 | 0.029366 | -0.016159 | 150 | 1 | 1.000000 | 1.000000 |
| strong utility action | 107 | 0.709 | 0.015110 | 0.028974 | -0.013863 | 107 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 107 | 0.709 | 0.015110 | 0.028974 | -0.013863 | 107 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 151 | 1.000 | 0.013207 | 0.029366 | -0.016159 | 150 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `31.0s`, rows `48`
- `32.5s` - `40.5s`, rows `17`
- `54.5s` - `75.0s`, rows `42`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `8.0`, LSTM `0.0034`, XGBoost `0.0298`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0035`, XGBoost `0.0296`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.0036`, XGBoost `0.0296`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.0040`, XGBoost `0.0298`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0043`, XGBoost `0.0295`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0060`, XGBoost `0.0307`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.0060`, XGBoost `0.0307`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.0291`, XGBoost `0.0539`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0062`, XGBoost `0.0307`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.0066`, XGBoost `0.0307`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
