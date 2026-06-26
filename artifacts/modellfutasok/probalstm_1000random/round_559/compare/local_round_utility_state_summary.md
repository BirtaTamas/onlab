# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `1`
- rows: `126`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 126 | 1.000 | 0.603795 | 0.687787 | -0.083992 | 2 | 124 | 0.730159 | 0.912698 |
| active/recent utility | 126 | 1.000 | 0.603795 | 0.687787 | -0.083992 | 2 | 124 | 0.730159 | 0.912698 |
| strong utility action | 59 | 0.468 | 0.485275 | 0.615610 | -0.130334 | 1 | 58 | 0.423729 | 0.813559 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 59 | 0.468 | 0.485275 | 0.615610 | -0.130334 | 1 | 58 | 0.423729 | 0.813559 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 126 | 1.000 | 0.603795 | 0.687787 | -0.083992 | 2 | 124 | 0.730159 | 0.912698 |

## Active Smoke/Inferno Intervals

- `16.5s` - `45.5s`, rows `59`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.0`, LSTM `0.3799`, XGBoost `0.6450`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `15.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.4002`, XGBoost `0.6557`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.6488`, XGBoost `0.9021`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.3957`, XGBoost `0.6450`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `15.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.4048`, XGBoost `0.6445`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `15.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.4160`, XGBoost `0.6557`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.4201`, XGBoost `0.6557`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.4204`, XGBoost `0.6551`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `15.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.4327`, XGBoost `0.6557`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.4227`, XGBoost `0.6456`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `15.0`, recent_utility `0`
