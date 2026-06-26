# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `2`
- rows: `202`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 202 | 1.000 | 0.587607 | 0.551863 | 0.035743 | 143 | 59 | 0.950495 | 0.336634 |
| active/recent utility | 202 | 1.000 | 0.587607 | 0.551863 | 0.035743 | 143 | 59 | 0.950495 | 0.336634 |
| strong utility action | 174 | 0.861 | 0.591304 | 0.539187 | 0.052117 | 136 | 38 | 1.000000 | 0.281609 |
| utility damage | 10 | 0.050 | 0.544290 | 0.480185 | 0.064105 | 10 | 0 | 1.000000 | 0.000000 |
| active smoke/inferno | 162 | 0.802 | 0.594575 | 0.542264 | 0.052311 | 124 | 38 | 1.000000 | 0.265432 |
| recent utility last 5s | 12 | 0.059 | 0.547146 | 0.497654 | 0.049493 | 12 | 0 | 1.000000 | 0.500000 |
| flash effect present | 202 | 1.000 | 0.587607 | 0.551863 | 0.035743 | 143 | 59 | 0.950495 | 0.336634 |

## Active Smoke/Inferno Intervals

- `6.5s` - `87.0s`, rows `162`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.5`, LSTM `0.5786`, XGBoost `0.4513`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.5442`, XGBoost `0.4218`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5656`, XGBoost `0.4513`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.5286`, XGBoost `0.4214`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.5784`, XGBoost `0.4716`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.5281`, XGBoost `0.4219`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.5607`, XGBoost `0.4585`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.5709`, XGBoost `0.4716`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5629`, XGBoost `0.4641`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.5633`, XGBoost `0.4646`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
