# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-the-mongolz-vs-natus-vincere-bo3-C0GZxMhpGHBr28LeyjgICZ/the-mongolz-vs-natus-vincere-m1-mirage.csv`
- round_num: `2`
- rows: `194`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 194 | 1.000 | 0.042571 | 0.061781 | -0.019210 | 180 | 14 | 1.000000 | 1.000000 |
| active/recent utility | 194 | 1.000 | 0.042571 | 0.061781 | -0.019210 | 180 | 14 | 1.000000 | 1.000000 |
| strong utility action | 131 | 0.675 | 0.054045 | 0.081201 | -0.027157 | 120 | 11 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 119 | 0.613 | 0.023573 | 0.060358 | -0.036786 | 119 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.103 | 0.286086 | 0.264085 | 0.022001 | 9 | 11 | 1.000000 | 1.000000 |
| flash effect present | 194 | 1.000 | 0.042571 | 0.061781 | -0.019210 | 180 | 14 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `66.5s`, rows `119`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `3.5`, LSTM `0.3963`, XGBoost `0.2879`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.3957`, XGBoost `0.2879`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.3900`, XGBoost `0.2867`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.3838`, XGBoost `0.2813`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.3688`, XGBoost `0.2813`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.5`, LSTM `0.2047`, XGBoost `0.2900`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `45.0`, LSTM `0.0052`, XGBoost `0.0857`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.0054`, XGBoost `0.0857`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0056`, XGBoost `0.0857`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.0049`, XGBoost `0.0849`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
