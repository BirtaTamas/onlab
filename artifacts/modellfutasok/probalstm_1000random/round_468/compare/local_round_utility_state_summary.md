# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv`
- round_num: `1`
- rows: `127`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 127 | 1.000 | 0.251171 | 0.330659 | -0.079488 | 124 | 3 | 0.897638 | 0.590551 |
| active/recent utility | 72 | 0.567 | 0.091863 | 0.185227 | -0.093364 | 71 | 1 | 1.000000 | 1.000000 |
| strong utility action | 48 | 0.378 | 0.106054 | 0.226378 | -0.120324 | 48 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 48 | 0.378 | 0.106054 | 0.226378 | -0.120324 | 48 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 72 | 0.567 | 0.091863 | 0.185227 | -0.093364 | 71 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `27.5s` - `51.0s`, rows `48`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.5`, LSTM `0.1149`, XGBoost `0.4186`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `84.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.1347`, XGBoost `0.4213`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `84.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.2025`, XGBoost `0.4544`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.1690`, XGBoost `0.4173`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `84.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.2120`, XGBoost `0.4526`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.2192`, XGBoost `0.4552`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.2277`, XGBoost `0.4503`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.2441`, XGBoost `0.4467`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0271`, XGBoost `0.1960`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `84.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.0306`, XGBoost `0.1987`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `84.0`, recent_utility `0`
