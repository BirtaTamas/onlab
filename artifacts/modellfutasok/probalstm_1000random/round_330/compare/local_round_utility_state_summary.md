# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `3`
- rows: `237`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 237 | 1.000 | 0.241504 | 0.376959 | -0.135456 | 219 | 18 | 0.890295 | 0.763713 |
| active/recent utility | 237 | 1.000 | 0.241504 | 0.376959 | -0.135456 | 219 | 18 | 0.890295 | 0.763713 |
| strong utility action | 116 | 0.489 | 0.287579 | 0.375771 | -0.088192 | 99 | 17 | 0.870690 | 0.724138 |
| utility damage | 10 | 0.042 | 0.082704 | 0.185323 | -0.102619 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 116 | 0.489 | 0.287579 | 0.375771 | -0.088192 | 99 | 17 | 0.870690 | 0.724138 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 237 | 1.000 | 0.241504 | 0.376959 | -0.135456 | 219 | 18 | 0.890295 | 0.763713 |

## Active Smoke/Inferno Intervals

- `9.0s` - `44.0s`, rows `71`
- `59.5s` - `81.5s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `70.0`, LSTM `0.4486`, XGBoost `0.6871`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.4585`, XGBoost `0.6871`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.4558`, XGBoost `0.6792`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.4669`, XGBoost `0.6871`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.4695`, XGBoost `0.6865`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.4702`, XGBoost `0.6855`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.4750`, XGBoost `0.6871`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.4682`, XGBoost `0.6779`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.4713`, XGBoost `0.6799`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.4784`, XGBoost `0.6787`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
