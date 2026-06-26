# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `6`
- rows: `263`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 263 | 1.000 | 0.306873 | 0.313203 | -0.006330 | 195 | 68 | 0.555133 | 0.612167 |
| active/recent utility | 263 | 1.000 | 0.306873 | 0.313203 | -0.006330 | 195 | 68 | 0.555133 | 0.612167 |
| strong utility action | 131 | 0.498 | 0.527698 | 0.537129 | -0.009431 | 88 | 43 | 0.221374 | 0.335878 |
| utility damage | 20 | 0.076 | 0.576970 | 0.599651 | -0.022681 | 20 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 131 | 0.498 | 0.527698 | 0.537129 | -0.009431 | 88 | 43 | 0.221374 | 0.335878 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 263 | 1.000 | 0.306873 | 0.313203 | -0.006330 | 195 | 68 | 0.555133 | 0.612167 |

## Active Smoke/Inferno Intervals

- `7.5s` - `72.5s`, rows `131`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.0`, LSTM `0.2765`, XGBoost `0.1046`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.5135`, XGBoost `0.3576`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.5056`, XGBoost `0.3578`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.3672`, XGBoost `0.4946`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.2306`, XGBoost `0.1046`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.3821`, XGBoost `0.4946`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.3797`, XGBoost `0.4878`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.3818`, XGBoost `0.4876`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.4642`, XGBoost `0.3610`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5043`, XGBoost `0.4018`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
