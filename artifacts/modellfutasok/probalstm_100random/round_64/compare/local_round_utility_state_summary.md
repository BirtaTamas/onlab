# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `12`
- rows: `225`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 225 | 1.000 | 0.188623 | 0.233242 | -0.044620 | 199 | 26 | 0.955556 | 0.977778 |
| active/recent utility | 225 | 1.000 | 0.188623 | 0.233242 | -0.044620 | 199 | 26 | 0.955556 | 0.977778 |
| strong utility action | 154 | 0.684 | 0.172517 | 0.229757 | -0.057240 | 148 | 6 | 1.000000 | 0.980519 |
| utility damage | 14 | 0.062 | 0.251603 | 0.295358 | -0.043755 | 14 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 154 | 0.684 | 0.172517 | 0.229757 | -0.057240 | 148 | 6 | 1.000000 | 0.980519 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 225 | 1.000 | 0.188623 | 0.233242 | -0.044620 | 199 | 26 | 0.955556 | 0.977778 |

## Active Smoke/Inferno Intervals

- `6.0s` - `53.5s`, rows `96`
- `59.0s` - `65.5s`, rows `14`
- `83.0s` - `104.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `8.0`, LSTM `0.0965`, XGBoost `0.2563`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1257`, XGBoost `0.2790`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.1025`, XGBoost `0.2558`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.1055`, XGBoost `0.2563`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.1153`, XGBoost `0.2653`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.1665`, XGBoost `0.3045`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1532`, XGBoost `0.2910`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.1738`, XGBoost `0.3045`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.1740`, XGBoost `0.3045`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.1657`, XGBoost `0.2939`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
