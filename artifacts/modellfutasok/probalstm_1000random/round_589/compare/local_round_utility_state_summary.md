# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `10`
- rows: `205`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 205 | 1.000 | 0.059472 | 0.103267 | -0.043795 | 197 | 8 | 1.000000 | 1.000000 |
| active/recent utility | 205 | 1.000 | 0.059472 | 0.103267 | -0.043795 | 197 | 8 | 1.000000 | 1.000000 |
| strong utility action | 185 | 0.902 | 0.059858 | 0.099840 | -0.039982 | 177 | 8 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 175 | 0.854 | 0.061528 | 0.098066 | -0.036538 | 167 | 8 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.049 | 0.030644 | 0.130893 | -0.100249 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 205 | 1.000 | 0.059472 | 0.103267 | -0.043795 | 197 | 8 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `33.0s`, rows `51`
- `40.5s` - `102.0s`, rows `124`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `6.0`, LSTM `0.0248`, XGBoost `0.1381`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.5`, LSTM `0.0259`, XGBoost `0.1381`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.0284`, XGBoost `0.1384`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.5`, LSTM `0.0293`, XGBoost `0.1384`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.0298`, XGBoost `0.1384`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `9.0`, LSTM `0.0329`, XGBoost `0.1384`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0381`, XGBoost `0.1370`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0417`, XGBoost `0.1384`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.0`, LSTM `0.0287`, XGBoost `0.1242`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.0325`, XGBoost `0.1278`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
