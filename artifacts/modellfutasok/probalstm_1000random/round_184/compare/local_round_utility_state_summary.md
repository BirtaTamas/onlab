# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-astralis-bo3-Zley6FZuKcttfrliAqsvWJ/astralis-vs-vitality-m1-inferno.csv`
- round_num: `15`
- rows: `169`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 169 | 1.000 | 0.078201 | 0.115803 | -0.037602 | 155 | 14 | 1.000000 | 1.000000 |
| active/recent utility | 169 | 1.000 | 0.078201 | 0.115803 | -0.037602 | 155 | 14 | 1.000000 | 1.000000 |
| strong utility action | 132 | 0.781 | 0.086997 | 0.118838 | -0.031841 | 118 | 14 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 132 | 0.781 | 0.086997 | 0.118838 | -0.031841 | 118 | 14 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 169 | 1.000 | 0.078201 | 0.115803 | -0.037602 | 155 | 14 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `19.0s`, rows `20`
- `20.0s` - `75.5s`, rows `112`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.0`, LSTM `0.1222`, XGBoost `0.2331`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.1140`, XGBoost `0.2196`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.0753`, XGBoost `0.1730`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.0907`, XGBoost `0.1845`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.0750`, XGBoost `0.1661`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.0767`, XGBoost `0.1663`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1038`, XGBoost `0.1927`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1046`, XGBoost `0.1931`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1048`, XGBoost `0.1927`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.1944`, XGBoost `0.2772`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
