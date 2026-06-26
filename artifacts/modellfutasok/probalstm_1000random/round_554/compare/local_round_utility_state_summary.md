# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `10`
- rows: `204`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 204 | 1.000 | 0.199793 | 0.364796 | -0.165004 | 204 | 0 | 1.000000 | 0.799020 |
| active/recent utility | 204 | 1.000 | 0.199793 | 0.364796 | -0.165004 | 204 | 0 | 1.000000 | 0.799020 |
| strong utility action | 182 | 0.892 | 0.193021 | 0.364239 | -0.171218 | 182 | 0 | 1.000000 | 0.829670 |
| utility damage | 21 | 0.103 | 0.245297 | 0.420217 | -0.174920 | 21 | 0 | 1.000000 | 0.523810 |
| active smoke/inferno | 162 | 0.794 | 0.183221 | 0.363506 | -0.180284 | 162 | 0 | 1.000000 | 0.870370 |
| recent utility last 5s | 10 | 0.049 | 0.143046 | 0.226080 | -0.083034 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 204 | 1.000 | 0.199793 | 0.364796 | -0.165004 | 204 | 0 | 1.000000 | 0.799020 |

## Active Smoke/Inferno Intervals

- `10.0s` - `90.5s`, rows `162`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `79.5`, LSTM `0.0958`, XGBoost `0.4602`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `46.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.1291`, XGBoost `0.4602`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `38.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.1192`, XGBoost `0.4267`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `11.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.1531`, XGBoost `0.4602`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `38.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.1184`, XGBoost `0.4255`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.1664`, XGBoost `0.4637`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.1182`, XGBoost `0.4113`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `41.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.1343`, XGBoost `0.4234`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `20.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.1484`, XGBoost `0.4339`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `11.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.1489`, XGBoost `0.4339`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `11.0`, recent_utility `0`
