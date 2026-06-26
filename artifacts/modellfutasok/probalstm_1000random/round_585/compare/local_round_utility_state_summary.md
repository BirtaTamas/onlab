# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m3-ancient.csv`
- round_num: `12`
- rows: `178`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 178 | 1.000 | 0.574594 | 0.526677 | 0.047918 | 161 | 17 | 0.977528 | 0.769663 |
| active/recent utility | 178 | 1.000 | 0.574594 | 0.526677 | 0.047918 | 161 | 17 | 0.977528 | 0.769663 |
| strong utility action | 155 | 0.871 | 0.575169 | 0.525675 | 0.049494 | 143 | 12 | 0.974194 | 0.741935 |
| utility damage | 10 | 0.056 | 0.551506 | 0.528812 | 0.022695 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 145 | 0.815 | 0.574506 | 0.525067 | 0.049439 | 133 | 12 | 0.972414 | 0.724138 |
| recent utility last 5s | 10 | 0.056 | 0.584782 | 0.534490 | 0.050292 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 178 | 1.000 | 0.574594 | 0.526677 | 0.047918 | 161 | 17 | 0.977528 | 0.769663 |

## Active Smoke/Inferno Intervals

- `7.5s` - `35.0s`, rows `56`
- `44.5s` - `88.5s`, rows `89`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.5`, LSTM `0.4501`, XGBoost `0.2043`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.4743`, XGBoost `0.2872`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.3483`, XGBoost `0.1913`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.6437`, XGBoost `0.5202`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.6148`, XGBoost `0.4924`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.6128`, XGBoost `0.4924`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.3027`, XGBoost `0.1913`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.6036`, XGBoost `0.4933`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.6025`, XGBoost `0.4931`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.8483`, XGBoost `0.7393`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
