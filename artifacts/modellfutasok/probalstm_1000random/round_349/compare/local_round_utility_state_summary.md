# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m3-ancient.csv`
- round_num: `12`
- rows: `215`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 215 | 1.000 | 0.682797 | 0.644215 | 0.038581 | 204 | 11 | 1.000000 | 1.000000 |
| active/recent utility | 215 | 1.000 | 0.682797 | 0.644215 | 0.038581 | 204 | 11 | 1.000000 | 1.000000 |
| strong utility action | 136 | 0.633 | 0.660561 | 0.629445 | 0.031116 | 128 | 8 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.047 | 0.625103 | 0.597799 | 0.027303 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 136 | 0.633 | 0.660561 | 0.629445 | 0.031116 | 128 | 8 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 215 | 1.000 | 0.682797 | 0.644215 | 0.038581 | 204 | 11 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `44.5s`, rows `77`
- `60.0s` - `82.0s`, rows `45`
- `99.0s` - `105.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `99.0`, LSTM `0.8756`, XGBoost `0.7869`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.8634`, XGBoost `0.7869`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.6937`, XGBoost `0.6271`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.6921`, XGBoost `0.6271`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.6459`, XGBoost `0.5852`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.6872`, XGBoost `0.6271`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.6860`, XGBoost `0.6275`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.6833`, XGBoost `0.6257`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.6833`, XGBoost `0.6257`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.6433`, XGBoost `0.5858`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
