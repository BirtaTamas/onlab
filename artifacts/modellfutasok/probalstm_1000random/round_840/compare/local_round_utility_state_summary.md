# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `2`
- rows: `243`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 243 | 1.000 | 0.244918 | 0.304970 | -0.060052 | 198 | 45 | 0.983539 | 0.971193 |
| active/recent utility | 243 | 1.000 | 0.244918 | 0.304970 | -0.060052 | 198 | 45 | 0.983539 | 0.971193 |
| strong utility action | 212 | 0.872 | 0.258330 | 0.322968 | -0.064638 | 174 | 38 | 0.981132 | 0.966981 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 202 | 0.831 | 0.253578 | 0.325001 | -0.071422 | 174 | 28 | 0.980198 | 0.965347 |
| recent utility last 5s | 20 | 0.082 | 0.275045 | 0.285032 | -0.009987 | 10 | 10 | 1.000000 | 1.000000 |
| flash effect present | 243 | 1.000 | 0.244918 | 0.304970 | -0.060052 | 198 | 45 | 0.983539 | 0.971193 |

## Active Smoke/Inferno Intervals

- `10.0s` - `33.0s`, rows `47`
- `34.5s` - `111.5s`, rows `155`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `98.5`, LSTM `0.1760`, XGBoost `0.4663`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.2011`, XGBoost `0.4663`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.2322`, XGBoost `0.4657`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.2064`, XGBoost `0.4393`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.2374`, XGBoost `0.4646`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.2177`, XGBoost `0.4404`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.2656`, XGBoost `0.4746`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.2295`, XGBoost `0.4363`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.2597`, XGBoost `0.4646`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.2590`, XGBoost `0.4617`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
