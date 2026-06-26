# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `8`
- rows: `215`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 215 | 1.000 | 0.575734 | 0.533182 | 0.042552 | 196 | 19 | 1.000000 | 0.795349 |
| active/recent utility | 215 | 1.000 | 0.575734 | 0.533182 | 0.042552 | 196 | 19 | 1.000000 | 0.795349 |
| strong utility action | 172 | 0.800 | 0.564982 | 0.521329 | 0.043653 | 163 | 9 | 1.000000 | 0.744186 |
| utility damage | 13 | 0.060 | 0.575136 | 0.510688 | 0.064448 | 13 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 172 | 0.800 | 0.564982 | 0.521329 | 0.043653 | 163 | 9 | 1.000000 | 0.744186 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 215 | 1.000 | 0.575734 | 0.533182 | 0.042552 | 196 | 19 | 1.000000 | 0.795349 |

## Active Smoke/Inferno Intervals

- `9.5s` - `34.5s`, rows `51`
- `39.0s` - `64.5s`, rows `52`
- `65.5s` - `99.5s`, rows `69`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `93.5`, LSTM `0.5509`, XGBoost `0.4662`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.5952`, XGBoost `0.5104`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.5359`, XGBoost `0.4566`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.5389`, XGBoost `0.4628`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.5992`, XGBoost `0.5233`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `106.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.5781`, XGBoost `0.5036`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `79.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.5373`, XGBoost `0.4638`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.5692`, XGBoost `0.4999`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.5690`, XGBoost `0.4999`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.5792`, XGBoost `0.5102`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `29.0`, recent_utility `0`
