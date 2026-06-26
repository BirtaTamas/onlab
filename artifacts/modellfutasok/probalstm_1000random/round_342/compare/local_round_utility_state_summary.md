# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `7`
- rows: `187`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 187 | 1.000 | 0.161060 | 0.210919 | -0.049859 | 183 | 4 | 1.000000 | 0.812834 |
| active/recent utility | 187 | 1.000 | 0.161060 | 0.210919 | -0.049859 | 183 | 4 | 1.000000 | 0.812834 |
| strong utility action | 142 | 0.759 | 0.133673 | 0.179766 | -0.046093 | 138 | 4 | 1.000000 | 0.887324 |
| utility damage | 17 | 0.091 | 0.267092 | 0.342495 | -0.075403 | 13 | 4 | 1.000000 | 1.000000 |
| active smoke/inferno | 142 | 0.759 | 0.133673 | 0.179766 | -0.046093 | 138 | 4 | 1.000000 | 0.887324 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 187 | 1.000 | 0.161060 | 0.210919 | -0.049859 | 183 | 4 | 1.000000 | 0.812834 |

## Active Smoke/Inferno Intervals

- `9.5s` - `32.5s`, rows `47`
- `39.5s` - `86.5s`, rows `95`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `45.0`, LSTM `0.3756`, XGBoost `0.5405`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.3288`, XGBoost `0.4909`, closer `lstm`, smoke `1`, inferno `4`, utility_damage `11.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.1557`, XGBoost `0.3175`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `157.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.0119`, XGBoost `0.1678`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.1236`, XGBoost `0.2758`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `92.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.1671`, XGBoost `0.3175`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `157.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.0174`, XGBoost `0.1666`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.1265`, XGBoost `0.2723`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `92.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.1680`, XGBoost `0.3136`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `149.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.3680`, XGBoost `0.5056`, closer `lstm`, smoke `1`, inferno `4`, utility_damage `3.0`, recent_utility `0`
