# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m2-mirage.csv`
- round_num: `2`
- rows: `156`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 156 | 1.000 | 0.176237 | 0.220143 | -0.043907 | 146 | 10 | 1.000000 | 1.000000 |
| active/recent utility | 156 | 1.000 | 0.176237 | 0.220143 | -0.043907 | 146 | 10 | 1.000000 | 1.000000 |
| strong utility action | 124 | 0.795 | 0.181370 | 0.219527 | -0.038156 | 114 | 10 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 124 | 0.795 | 0.181370 | 0.219527 | -0.038156 | 114 | 10 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 156 | 1.000 | 0.176237 | 0.220143 | -0.043907 | 146 | 10 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `34.0s`, rows `52`
- `40.5s` - `76.0s`, rows `72`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.0`, LSTM `0.1575`, XGBoost `0.2592`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.0719`, XGBoost `0.1727`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.1579`, XGBoost `0.2568`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.0638`, XGBoost `0.1593`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.1421`, XGBoost `0.2364`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.0750`, XGBoost `0.1680`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.1717`, XGBoost `0.2594`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.1735`, XGBoost `0.2598`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.1697`, XGBoost `0.2547`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.1761`, XGBoost `0.2581`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
