# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-lynn-vision-vs-furia-bo3-RhNzrLTGYeGsl1rd1jweWL/lynn-vision-vs-furia-m2-anubis.csv`
- round_num: `18`
- rows: `183`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 183 | 1.000 | 0.066354 | 0.086691 | -0.020337 | 182 | 1 | 1.000000 | 0.885246 |
| active/recent utility | 183 | 1.000 | 0.066354 | 0.086691 | -0.020337 | 182 | 1 | 1.000000 | 0.885246 |
| strong utility action | 110 | 0.601 | 0.047328 | 0.067690 | -0.020362 | 109 | 1 | 1.000000 | 0.945455 |
| utility damage | 13 | 0.071 | 0.183991 | 0.243425 | -0.059434 | 12 | 1 | 1.000000 | 0.923077 |
| active smoke/inferno | 110 | 0.601 | 0.047328 | 0.067690 | -0.020362 | 109 | 1 | 1.000000 | 0.945455 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 183 | 1.000 | 0.066354 | 0.086691 | -0.020337 | 182 | 1 | 1.000000 | 0.885246 |

## Active Smoke/Inferno Intervals

- `7.5s` - `37.5s`, rows `61`
- `39.0s` - `63.0s`, rows `49`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.5`, LSTM `0.0519`, XGBoost `0.1448`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.0529`, XGBoost `0.1441`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0539`, XGBoost `0.1450`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.0540`, XGBoost `0.1450`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.0548`, XGBoost `0.1448`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.2142`, XGBoost `0.2980`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.0430`, XGBoost `0.1180`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0435`, XGBoost `0.1177`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0632`, XGBoost `0.1355`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.0452`, XGBoost `0.1163`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
