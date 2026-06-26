# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m3-ancient.csv`
- round_num: `16`
- rows: `164`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 164 | 1.000 | 0.104009 | 0.118368 | -0.014359 | 153 | 11 | 0.945122 | 0.853659 |
| active/recent utility | 164 | 1.000 | 0.104009 | 0.118368 | -0.014359 | 153 | 11 | 0.945122 | 0.853659 |
| strong utility action | 86 | 0.524 | 0.121662 | 0.135183 | -0.013521 | 75 | 11 | 0.895349 | 0.872093 |
| utility damage | 20 | 0.122 | 0.395880 | 0.405108 | -0.009228 | 13 | 7 | 0.600000 | 0.650000 |
| active smoke/inferno | 86 | 0.524 | 0.121662 | 0.135183 | -0.013521 | 75 | 11 | 0.895349 | 0.872093 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 164 | 1.000 | 0.104009 | 0.118368 | -0.014359 | 153 | 11 | 0.945122 | 0.853659 |

## Active Smoke/Inferno Intervals

- `6.5s` - `49.0s`, rows `86`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.5`, LSTM `0.2846`, XGBoost `0.3384`, closer `lstm`, smoke `6`, inferno `2`, utility_damage `26.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.2961`, XGBoost `0.3379`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.4931`, XGBoost `0.5304`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.5391`, XGBoost `0.5035`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `67.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.5242`, XGBoost `0.4886`, closer `xgboost`, smoke `6`, inferno `2`, utility_damage `67.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0039`, XGBoost `0.0383`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.0039`, XGBoost `0.0383`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.0040`, XGBoost `0.0383`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.0038`, XGBoost `0.0369`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0038`, XGBoost `0.0365`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
