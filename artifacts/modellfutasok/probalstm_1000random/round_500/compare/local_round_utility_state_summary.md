# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m2-dust2.csv`
- round_num: `3`
- rows: `243`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 243 | 1.000 | 0.069855 | 0.171506 | -0.101651 | 243 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 243 | 1.000 | 0.069855 | 0.171506 | -0.101651 | 243 | 0 | 1.000000 | 1.000000 |
| strong utility action | 162 | 0.667 | 0.072494 | 0.211501 | -0.139007 | 162 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 162 | 0.667 | 0.072494 | 0.211501 | -0.139007 | 162 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.041 | 0.075720 | 0.275143 | -0.199423 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 243 | 1.000 | 0.069855 | 0.171506 | -0.101651 | 243 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `50.0s`, rows `87`
- `52.5s` - `89.5s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `68.5`, LSTM `0.0283`, XGBoost `0.2566`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.0314`, XGBoost `0.2571`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.0322`, XGBoost `0.2543`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.0414`, XGBoost `0.2615`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.0597`, XGBoost `0.2797`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.0588`, XGBoost `0.2779`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.0364`, XGBoost `0.2548`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.0377`, XGBoost `0.2548`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0651`, XGBoost `0.2787`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0553`, XGBoost `0.2674`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
