# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-natus-vincere-vs-3dmax-bo3-JB3JZO-5zNCohi5tAgyHtq/natus-vincere-vs-3dmax-m2-inferno.csv`
- round_num: `4`
- rows: `195`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 195 | 1.000 | 0.073519 | 0.109822 | -0.036303 | 169 | 26 | 1.000000 | 1.000000 |
| active/recent utility | 195 | 1.000 | 0.073519 | 0.109822 | -0.036303 | 169 | 26 | 1.000000 | 1.000000 |
| strong utility action | 127 | 0.651 | 0.061602 | 0.073783 | -0.012181 | 101 | 26 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 127 | 0.651 | 0.061602 | 0.073783 | -0.012181 | 101 | 26 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 195 | 1.000 | 0.073519 | 0.109822 | -0.036303 | 169 | 26 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `17.5s`, rows `16`
- `33.5s` - `88.5s`, rows `111`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `17.5`, LSTM `0.1139`, XGBoost `0.2358`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `4.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.1352`, XGBoost `0.2521`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.1215`, XGBoost `0.2355`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `4.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.1305`, XGBoost `0.2434`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1409`, XGBoost `0.2526`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.1409`, XGBoost `0.2518`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1420`, XGBoost `0.2510`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.1266`, XGBoost `0.2340`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1464`, XGBoost `0.2518`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1442`, XGBoost `0.2473`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
