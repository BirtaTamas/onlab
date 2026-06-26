# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3/the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `3`
- rows: `145`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 145 | 1.000 | 0.162632 | 0.221342 | -0.058710 | 139 | 6 | 1.000000 | 0.986207 |
| active/recent utility | 145 | 1.000 | 0.162632 | 0.221342 | -0.058710 | 139 | 6 | 1.000000 | 0.986207 |
| strong utility action | 116 | 0.800 | 0.181568 | 0.250252 | -0.068684 | 110 | 6 | 1.000000 | 0.982759 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 105 | 0.724 | 0.160156 | 0.225817 | -0.065662 | 99 | 6 | 1.000000 | 0.980952 |
| recent utility last 5s | 11 | 0.076 | 0.385953 | 0.483486 | -0.097533 | 11 | 0 | 1.000000 | 1.000000 |
| flash effect present | 145 | 1.000 | 0.162632 | 0.221342 | -0.058710 | 139 | 6 | 1.000000 | 0.986207 |

## Active Smoke/Inferno Intervals

- `8.5s` - `60.5s`, rows `105`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.5`, LSTM `0.1322`, XGBoost `0.4319`, closer `lstm`, smoke `6`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.1477`, XGBoost `0.4319`, closer `lstm`, smoke `6`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.2166`, XGBoost `0.4678`, closer `lstm`, smoke `6`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.1187`, XGBoost `0.3680`, closer `lstm`, smoke `7`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.1868`, XGBoost `0.4311`, closer `lstm`, smoke `7`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.1254`, XGBoost `0.3663`, closer `lstm`, smoke `7`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.1986`, XGBoost `0.4311`, closer `lstm`, smoke `7`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.3573`, XGBoost `0.5754`, closer `lstm`, smoke `7`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.1573`, XGBoost `0.3653`, closer `lstm`, smoke `7`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.2777`, XGBoost `0.4653`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
