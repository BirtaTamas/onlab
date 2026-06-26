# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m1-inferno.csv`
- round_num: `2`
- rows: `246`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 246 | 1.000 | 0.154064 | 0.210648 | -0.056585 | 232 | 14 | 1.000000 | 1.000000 |
| active/recent utility | 246 | 1.000 | 0.154064 | 0.210648 | -0.056585 | 232 | 14 | 1.000000 | 1.000000 |
| strong utility action | 190 | 0.772 | 0.154788 | 0.220851 | -0.066062 | 183 | 7 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.041 | 0.310400 | 0.383409 | -0.073009 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 188 | 0.764 | 0.153095 | 0.219138 | -0.066043 | 181 | 7 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 246 | 1.000 | 0.154064 | 0.210648 | -0.056585 | 232 | 14 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `17.5s`, rows `14`
- `21.5s` - `108.0s`, rows `174`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `45.5`, LSTM `0.1430`, XGBoost `0.3447`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.1447`, XGBoost `0.3454`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.1462`, XGBoost `0.3468`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.1453`, XGBoost `0.3454`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.1463`, XGBoost `0.3454`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.1542`, XGBoost `0.3468`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.1560`, XGBoost `0.3475`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.1744`, XGBoost `0.3572`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.1753`, XGBoost `0.3574`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.1720`, XGBoost `0.3468`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
