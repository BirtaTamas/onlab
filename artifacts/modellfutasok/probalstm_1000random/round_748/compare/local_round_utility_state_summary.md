# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `7`
- rows: `198`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 198 | 1.000 | 0.680592 | 0.678126 | 0.002466 | 103 | 95 | 1.000000 | 1.000000 |
| active/recent utility | 198 | 1.000 | 0.680592 | 0.678126 | 0.002466 | 103 | 95 | 1.000000 | 1.000000 |
| strong utility action | 196 | 0.990 | 0.680247 | 0.678767 | 0.001481 | 101 | 95 | 1.000000 | 1.000000 |
| utility damage | 13 | 0.066 | 0.674912 | 0.619711 | 0.055202 | 13 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 184 | 0.929 | 0.681907 | 0.683900 | -0.001993 | 89 | 95 | 1.000000 | 1.000000 |
| recent utility last 5s | 13 | 0.066 | 0.651428 | 0.599275 | 0.052153 | 13 | 0 | 1.000000 | 1.000000 |
| flash effect present | 198 | 1.000 | 0.680592 | 0.678126 | 0.002466 | 103 | 95 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `98.5s`, rows `184`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `95.0`, LSTM `0.9029`, XGBoost `0.7907`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `1.0`, LSTM `0.7099`, XGBoost `0.6030`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.7054`, XGBoost `0.6045`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `95.5`, LSTM `0.8880`, XGBoost `0.7907`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.0`, LSTM `0.6953`, XGBoost `0.6045`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `96.0`, LSTM `0.8761`, XGBoost `0.7907`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.5616`, XGBoost `0.6383`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.5632`, XGBoost `0.6383`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.6656`, XGBoost `0.7401`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `17.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.5641`, XGBoost `0.6383`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
