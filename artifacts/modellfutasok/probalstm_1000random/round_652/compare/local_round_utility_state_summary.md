# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-eternal-fire-vs-spirit-bo5-7H36TpK_LYGHtCXpF3Cgdr/eternal-fire-vs-spirit-m3-dust2.csv`
- round_num: `5`
- rows: `163`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 163 | 1.000 | 0.582413 | 0.561225 | 0.021188 | 106 | 57 | 0.736196 | 0.361963 |
| active/recent utility | 163 | 1.000 | 0.582413 | 0.561225 | 0.021188 | 106 | 57 | 0.736196 | 0.361963 |
| strong utility action | 148 | 0.908 | 0.562895 | 0.542592 | 0.020303 | 93 | 55 | 0.709459 | 0.324324 |
| utility damage | 10 | 0.061 | 0.710129 | 0.623512 | 0.086617 | 10 | 0 | 0.800000 | 0.800000 |
| active smoke/inferno | 148 | 0.908 | 0.562895 | 0.542592 | 0.020303 | 93 | 55 | 0.709459 | 0.324324 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 163 | 1.000 | 0.582413 | 0.561225 | 0.021188 | 106 | 57 | 0.736196 | 0.361963 |

## Active Smoke/Inferno Intervals

- `2.5s` - `76.0s`, rows `148`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.5`, LSTM `0.3501`, XGBoost `0.1847`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.8796`, XGBoost `0.7507`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.8749`, XGBoost `0.7552`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.3410`, XGBoost `0.2217`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.8695`, XGBoost `0.7558`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.3093`, XGBoost `0.1970`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.8660`, XGBoost `0.7552`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.2887`, XGBoost `0.1820`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.8569`, XGBoost `0.7507`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.5669`, XGBoost `0.4685`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `27.0`, recent_utility `0`
