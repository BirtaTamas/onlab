# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv`
- round_num: `3`
- rows: `104`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 104 | 1.000 | 0.575127 | 0.520662 | 0.054465 | 99 | 5 | 0.971154 | 0.298077 |
| active/recent utility | 104 | 1.000 | 0.575127 | 0.520662 | 0.054465 | 99 | 5 | 0.971154 | 0.298077 |
| strong utility action | 92 | 0.885 | 0.580363 | 0.524024 | 0.056340 | 87 | 5 | 0.967391 | 0.326087 |
| utility damage | 15 | 0.144 | 0.540260 | 0.469089 | 0.071171 | 12 | 3 | 0.800000 | 0.333333 |
| active smoke/inferno | 92 | 0.885 | 0.580363 | 0.524024 | 0.056340 | 87 | 5 | 0.967391 | 0.326087 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 104 | 1.000 | 0.575127 | 0.520662 | 0.054465 | 99 | 5 | 0.971154 | 0.298077 |

## Active Smoke/Inferno Intervals

- `6.0s` - `51.5s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.5`, LSTM `0.4411`, XGBoost `0.2506`, closer `lstm`, smoke `2`, inferno `4`, utility_damage `8.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.6288`, XGBoost `0.7738`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `16.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.3984`, XGBoost `0.2651`, closer `lstm`, smoke `1`, inferno `4`, utility_damage `20.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.5821`, XGBoost `0.4491`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `8.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5787`, XGBoost `0.4489`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `8.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5699`, XGBoost `0.4458`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `8.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5691`, XGBoost `0.4464`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `8.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5701`, XGBoost `0.4539`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `8.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5635`, XGBoost `0.4489`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `8.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.3549`, XGBoost `0.2502`, closer `lstm`, smoke `1`, inferno `4`, utility_damage `12.0`, recent_utility `0`
