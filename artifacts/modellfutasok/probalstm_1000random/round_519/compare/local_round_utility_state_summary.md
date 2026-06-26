# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `7`
- rows: `275`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 275 | 1.000 | 0.372155 | 0.392426 | -0.020271 | 176 | 99 | 0.574545 | 0.567273 |
| active/recent utility | 275 | 1.000 | 0.372155 | 0.392426 | -0.020271 | 176 | 99 | 0.574545 | 0.567273 |
| strong utility action | 189 | 0.687 | 0.516439 | 0.516169 | 0.000270 | 92 | 97 | 0.380952 | 0.370370 |
| utility damage | 10 | 0.036 | 0.276451 | 0.450590 | -0.174139 | 10 | 0 | 1.000000 | 0.900000 |
| active smoke/inferno | 189 | 0.687 | 0.516439 | 0.516169 | 0.000270 | 92 | 97 | 0.380952 | 0.370370 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 275 | 1.000 | 0.372155 | 0.392426 | -0.020271 | 176 | 99 | 0.574545 | 0.567273 |

## Active Smoke/Inferno Intervals

- `10.5s` - `104.5s`, rows `189`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.0`, LSTM `0.1961`, XGBoost `0.4298`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.2028`, XGBoost `0.4259`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.2092`, XGBoost `0.4305`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `17.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.2099`, XGBoost `0.4259`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.2152`, XGBoost `0.4300`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `17.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.4727`, XGBoost `0.2646`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.2168`, XGBoost `0.4219`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.4687`, XGBoost `0.2646`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.4519`, XGBoost `0.2595`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.2308`, XGBoost `0.4221`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
