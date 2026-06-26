# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `16`
- rows: `252`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 252 | 1.000 | 0.598422 | 0.645958 | -0.047536 | 189 | 63 | 0.115079 | 0.154762 |
| active/recent utility | 252 | 1.000 | 0.598422 | 0.645958 | -0.047536 | 189 | 63 | 0.115079 | 0.154762 |
| strong utility action | 154 | 0.611 | 0.637395 | 0.734061 | -0.096665 | 147 | 7 | 0.000000 | 0.000000 |
| utility damage | 13 | 0.052 | 0.582847 | 0.646241 | -0.063394 | 13 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 154 | 0.611 | 0.637395 | 0.734061 | -0.096665 | 147 | 7 | 0.000000 | 0.000000 |
| recent utility last 5s | 10 | 0.040 | 0.592199 | 0.788704 | -0.196505 | 10 | 0 | 0.000000 | 0.000000 |
| flash effect present | 252 | 1.000 | 0.598422 | 0.645958 | -0.047536 | 189 | 63 | 0.115079 | 0.154762 |

## Active Smoke/Inferno Intervals

- `8.0s` - `58.5s`, rows `102`
- `65.0s` - `90.5s`, rows `52`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.5`, LSTM `0.5836`, XGBoost `0.7892`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `27.0`, LSTM `0.5851`, XGBoost `0.7881`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `29.5`, LSTM `0.5911`, XGBoost `0.7899`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `30.0`, LSTM `0.5916`, XGBoost `0.7885`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `28.0`, LSTM `0.5917`, XGBoost `0.7885`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `29.0`, LSTM `0.5918`, XGBoost `0.7881`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `27.5`, LSTM `0.5930`, XGBoost `0.7885`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `31.0`, LSTM `0.5942`, XGBoost `0.7888`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `28.5`, LSTM `0.5951`, XGBoost `0.7881`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `26.5`, LSTM `0.5959`, XGBoost `0.7881`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
