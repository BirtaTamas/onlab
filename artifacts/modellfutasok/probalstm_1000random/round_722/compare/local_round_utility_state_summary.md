# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `12`
- rows: `211`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 211 | 1.000 | 0.214566 | 0.260209 | -0.045643 | 150 | 61 | 0.886256 | 1.000000 |
| active/recent utility | 211 | 1.000 | 0.214566 | 0.260209 | -0.045643 | 150 | 61 | 0.886256 | 1.000000 |
| strong utility action | 158 | 0.749 | 0.219306 | 0.243912 | -0.024606 | 101 | 57 | 0.848101 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 158 | 0.749 | 0.219306 | 0.243912 | -0.024606 | 101 | 57 | 0.848101 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 211 | 1.000 | 0.214566 | 0.260209 | -0.045643 | 150 | 61 | 0.886256 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `64.5s`, rows `113`
- `78.5s` - `100.5s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `79.0`, LSTM `0.0630`, XGBoost `0.3861`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `100.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.0732`, XGBoost `0.3861`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `100.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.0783`, XGBoost `0.3267`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0941`, XGBoost `0.3267`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.1053`, XGBoost `0.3267`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.1223`, XGBoost `0.3267`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.0301`, XGBoost `0.2245`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.0295`, XGBoost `0.2232`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.1489`, XGBoost `0.3407`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0325`, XGBoost `0.2231`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
