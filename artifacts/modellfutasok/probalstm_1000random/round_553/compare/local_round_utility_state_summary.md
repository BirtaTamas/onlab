# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `11`
- rows: `183`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 183 | 1.000 | 0.135638 | 0.191884 | -0.056245 | 182 | 1 | 0.945355 | 0.939891 |
| active/recent utility | 183 | 1.000 | 0.135638 | 0.191884 | -0.056245 | 182 | 1 | 0.945355 | 0.939891 |
| strong utility action | 129 | 0.705 | 0.188289 | 0.266075 | -0.077787 | 128 | 1 | 0.922481 | 0.914729 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 118 | 0.645 | 0.164004 | 0.245190 | -0.081186 | 118 | 0 | 0.915254 | 0.906780 |
| recent utility last 5s | 17 | 0.093 | 0.442969 | 0.490006 | -0.047036 | 16 | 1 | 1.000000 | 1.000000 |
| flash effect present | 183 | 1.000 | 0.135638 | 0.191884 | -0.056245 | 182 | 1 | 0.945355 | 0.939891 |

## Active Smoke/Inferno Intervals

- `6.0s` - `64.5s`, rows `118`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.0`, LSTM `0.1715`, XGBoost `0.4184`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.4879`, XGBoost `0.7301`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.2485`, XGBoost `0.4563`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5051`, XGBoost `0.7011`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0483`, XGBoost `0.2437`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.0549`, XGBoost `0.2475`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5068`, XGBoost `0.6989`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0524`, XGBoost `0.2404`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0533`, XGBoost `0.2404`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0638`, XGBoost `0.2497`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
