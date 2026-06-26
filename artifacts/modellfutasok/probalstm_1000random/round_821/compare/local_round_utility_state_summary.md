# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `13`
- rows: `120`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 120 | 1.000 | 0.126881 | 0.133029 | -0.006147 | 92 | 28 | 0.775000 | 1.000000 |
| active/recent utility | 120 | 1.000 | 0.126881 | 0.133029 | -0.006147 | 92 | 28 | 0.775000 | 1.000000 |
| strong utility action | 50 | 0.417 | 0.004372 | 0.015966 | -0.011594 | 50 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 50 | 0.417 | 0.004372 | 0.015966 | -0.011594 | 50 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 120 | 1.000 | 0.126881 | 0.133029 | -0.006147 | 92 | 28 | 0.775000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `24.5s` - `49.0s`, rows `50`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `26.5`, LSTM `0.0380`, XGBoost `0.1079`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0065`, XGBoost `0.0292`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.0069`, XGBoost `0.0284`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.0081`, XGBoost `0.0280`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0093`, XGBoost `0.0290`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.0053`, XGBoost `0.0242`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0108`, XGBoost `0.0293`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0055`, XGBoost `0.0236`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0070`, XGBoost `0.0248`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.0087`, XGBoost `0.0258`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
