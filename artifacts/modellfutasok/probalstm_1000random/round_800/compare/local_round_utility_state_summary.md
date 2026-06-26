# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `3`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.922687 | 0.959076 | -0.036389 | 4 | 226 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.922687 | 0.959076 | -0.036389 | 4 | 226 | 1.000000 | 1.000000 |
| strong utility action | 164 | 0.713 | 0.919209 | 0.956212 | -0.037002 | 4 | 160 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.043 | 0.851970 | 0.912402 | -0.060431 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 154 | 0.670 | 0.924836 | 0.961570 | -0.036734 | 4 | 150 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.043 | 0.832563 | 0.873695 | -0.041132 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.922687 | 0.959076 | -0.036389 | 4 | 226 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `84.0s`, rows `154`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.5`, LSTM `0.7865`, XGBoost `0.9144`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.8098`, XGBoost `0.9199`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.8065`, XGBoost `0.9147`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.8292`, XGBoost `0.9148`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.8350`, XGBoost `0.9143`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.8376`, XGBoost `0.9144`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.8377`, XGBoost `0.9143`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.8381`, XGBoost `0.9144`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.8361`, XGBoost `0.9114`, closer `xgboost`, smoke `1`, inferno `4`, utility_damage `33.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.8362`, XGBoost `0.9113`, closer `xgboost`, smoke `1`, inferno `4`, utility_damage `33.0`, recent_utility `0`
