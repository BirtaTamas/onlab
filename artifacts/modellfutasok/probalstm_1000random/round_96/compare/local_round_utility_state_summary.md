# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `14`
- rows: `141`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 141 | 1.000 | 0.343707 | 0.464319 | -0.120613 | 141 | 0 | 0.702128 | 0.638298 |
| active/recent utility | 141 | 1.000 | 0.343707 | 0.464319 | -0.120613 | 141 | 0 | 0.702128 | 0.638298 |
| strong utility action | 103 | 0.730 | 0.395086 | 0.501828 | -0.106742 | 103 | 0 | 0.631068 | 0.553398 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 103 | 0.730 | 0.395086 | 0.501828 | -0.106742 | 103 | 0 | 0.631068 | 0.553398 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 141 | 1.000 | 0.343707 | 0.464319 | -0.120613 | 141 | 0 | 0.702128 | 0.638298 |

## Active Smoke/Inferno Intervals

- `8.5s` - `30.5s`, rows `45`
- `37.5s` - `44.0s`, rows `14`
- `47.0s` - `68.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `8.5`, LSTM `0.1705`, XGBoost `0.3987`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.4131`, XGBoost `0.6226`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1926`, XGBoost `0.4005`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.1964`, XGBoost `0.3999`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.2133`, XGBoost `0.4005`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.1280`, XGBoost `0.3140`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `46.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.1316`, XGBoost `0.3140`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.1387`, XGBoost `0.3140`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.2253`, XGBoost `0.4005`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.3272`, XGBoost `0.4995`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
