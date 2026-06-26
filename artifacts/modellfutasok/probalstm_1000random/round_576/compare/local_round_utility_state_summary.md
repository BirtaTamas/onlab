# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `3`
- rows: `140`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 140 | 1.000 | 0.003620 | 0.011681 | -0.008061 | 131 | 9 | 1.000000 | 1.000000 |
| active/recent utility | 140 | 1.000 | 0.003620 | 0.011681 | -0.008061 | 131 | 9 | 1.000000 | 1.000000 |
| strong utility action | 47 | 0.336 | 0.005105 | 0.011982 | -0.006877 | 42 | 5 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 47 | 0.336 | 0.005105 | 0.011982 | -0.006877 | 42 | 5 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 140 | 1.000 | 0.003620 | 0.011681 | -0.008061 | 131 | 9 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `30.5s`, rows `47`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `7.5`, LSTM `0.0094`, XGBoost `0.0633`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0099`, XGBoost `0.0636`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0121`, XGBoost `0.0645`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0093`, XGBoost `0.0388`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `102.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0103`, XGBoost `0.0393`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `102.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.0099`, XGBoost `0.0381`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `102.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0095`, XGBoost `0.0236`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `102.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0067`, XGBoost `0.0162`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `102.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0094`, XGBoost `0.0167`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `102.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0082`, XGBoost `0.0148`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `102.0`, recent_utility `0`
