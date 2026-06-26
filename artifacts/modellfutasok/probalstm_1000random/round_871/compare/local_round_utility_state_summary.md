# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-tyloo-vs-vitality-bo3-aF98ikh3PjdqKlkdIJn9tC/tyloo-vs-vitality-m1-inferno.csv`
- round_num: `8`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.723241 | 0.682997 | 0.040244 | 139 | 91 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.723241 | 0.682997 | 0.040244 | 139 | 91 | 1.000000 | 1.000000 |
| strong utility action | 153 | 0.665 | 0.682677 | 0.632758 | 0.049920 | 104 | 49 | 1.000000 | 1.000000 |
| utility damage | 41 | 0.178 | 0.637630 | 0.574389 | 0.063241 | 33 | 8 | 1.000000 | 1.000000 |
| active smoke/inferno | 150 | 0.652 | 0.684174 | 0.634925 | 0.049249 | 101 | 49 | 1.000000 | 1.000000 |
| recent utility last 5s | 21 | 0.091 | 0.608357 | 0.509674 | 0.098683 | 21 | 0 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.723241 | 0.682997 | 0.040244 | 139 | 91 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.5s` - `27.0s`, rows `32`
- `34.5s` - `93.0s`, rows `118`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.0`, LSTM `0.6138`, XGBoost `0.5003`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `43.5`, LSTM `0.6151`, XGBoost `0.5024`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `15.0`, LSTM `0.6293`, XGBoost `0.5176`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `11.0`, LSTM `0.6280`, XGBoost `0.5166`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `7.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.6178`, XGBoost `0.5068`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `31.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6248`, XGBoost `0.5170`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `7.0`, recent_utility `1`
- seconds `23.0`, LSTM `0.6311`, XGBoost `0.5237`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `63.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.6079`, XGBoost `0.5009`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `13.5`, LSTM `0.6203`, XGBoost `0.5149`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `7.0`, recent_utility `1`
- seconds `14.0`, LSTM `0.6222`, XGBoost `0.5170`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `7.0`, recent_utility `1`
