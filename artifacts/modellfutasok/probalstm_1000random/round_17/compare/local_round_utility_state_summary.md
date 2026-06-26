# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `4`
- rows: `198`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 198 | 1.000 | 0.012214 | 0.029578 | -0.017364 | 195 | 3 | 1.000000 | 1.000000 |
| active/recent utility | 198 | 1.000 | 0.012214 | 0.029578 | -0.017364 | 195 | 3 | 1.000000 | 1.000000 |
| strong utility action | 158 | 0.798 | 0.008266 | 0.022119 | -0.013853 | 155 | 3 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.051 | 0.052163 | 0.116203 | -0.064040 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 158 | 0.798 | 0.008266 | 0.022119 | -0.013853 | 155 | 3 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 198 | 1.000 | 0.012214 | 0.029578 | -0.017364 | 195 | 3 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `19.5s`, rows `21`
- `21.5s` - `28.0s`, rows `14`
- `30.5s` - `69.5s`, rows `79`
- `76.5s` - `98.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `21.5`, LSTM `0.0228`, XGBoost `0.1091`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `163.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.0233`, XGBoost `0.1057`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `165.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.0472`, XGBoost `0.1168`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `27.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.0471`, XGBoost `0.1154`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `27.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0477`, XGBoost `0.1156`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `27.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.0493`, XGBoost `0.1168`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `27.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0503`, XGBoost `0.1159`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `27.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0537`, XGBoost `0.1154`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `27.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.0555`, XGBoost `0.1168`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `27.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.0560`, XGBoost `0.1171`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `27.0`, recent_utility `0`
