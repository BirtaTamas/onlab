# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `5`
- rows: `274`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 274 | 1.000 | 0.265598 | 0.393086 | -0.127489 | 258 | 16 | 0.908759 | 0.503650 |
| active/recent utility | 274 | 1.000 | 0.265598 | 0.393086 | -0.127489 | 258 | 16 | 0.908759 | 0.503650 |
| strong utility action | 170 | 0.620 | 0.206560 | 0.325169 | -0.118609 | 167 | 3 | 0.994118 | 0.594118 |
| utility damage | 10 | 0.036 | 0.014991 | 0.151585 | -0.136593 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 167 | 0.609 | 0.210027 | 0.328365 | -0.118338 | 164 | 3 | 0.994012 | 0.586826 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 274 | 1.000 | 0.265598 | 0.393086 | -0.127489 | 258 | 16 | 0.908759 | 0.503650 |

## Active Smoke/Inferno Intervals

- `6.0s` - `63.0s`, rows `115`
- `91.0s` - `97.5s`, rows `14`
- `118.0s` - `136.5s`, rows `38`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `92.5`, LSTM `0.1597`, XGBoost `0.5850`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.1523`, XGBoost `0.5628`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.1680`, XGBoost `0.5656`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.1760`, XGBoost `0.5628`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.1356`, XGBoost `0.4453`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.1389`, XGBoost `0.4453`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.1515`, XGBoost `0.4453`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.1529`, XGBoost `0.4453`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.1597`, XGBoost `0.4453`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.3091`, XGBoost `0.5909`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
