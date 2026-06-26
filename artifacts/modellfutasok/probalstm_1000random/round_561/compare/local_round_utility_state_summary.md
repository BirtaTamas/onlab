# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-spirit-vs-astralis-bo3-GZVTrKsE-zdG9dH6juITei/spirit-vs-astralis-m1-nuke.csv`
- round_num: `3`
- rows: `296`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 296 | 1.000 | 0.047016 | 0.137118 | -0.090102 | 276 | 20 | 1.000000 | 1.000000 |
| active/recent utility | 296 | 1.000 | 0.047016 | 0.137118 | -0.090102 | 276 | 20 | 1.000000 | 1.000000 |
| strong utility action | 181 | 0.611 | 0.062303 | 0.178464 | -0.116161 | 180 | 1 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.034 | 0.175337 | 0.209221 | -0.033883 | 9 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 181 | 0.611 | 0.062303 | 0.178464 | -0.116161 | 180 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 296 | 1.000 | 0.047016 | 0.137118 | -0.090102 | 276 | 20 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `69.0s`, rows `119`
- `79.0s` - `85.5s`, rows `14`
- `90.5s` - `114.0s`, rows `48`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.5`, LSTM `0.1246`, XGBoost `0.3360`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.1416`, XGBoost `0.3360`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.0256`, XGBoost `0.2186`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.0287`, XGBoost `0.2191`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.0359`, XGBoost `0.2237`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.0495`, XGBoost `0.2342`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.0417`, XGBoost `0.2237`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.0545`, XGBoost `0.2340`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.0547`, XGBoost `0.2340`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.0166`, XGBoost `0.1954`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
