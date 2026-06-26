# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `7`
- rows: `206`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 206 | 1.000 | 0.362066 | 0.422756 | -0.060690 | 73 | 133 | 0.131068 | 0.169903 |
| active/recent utility | 206 | 1.000 | 0.362066 | 0.422756 | -0.060690 | 73 | 133 | 0.131068 | 0.169903 |
| strong utility action | 166 | 0.806 | 0.360249 | 0.408126 | -0.047877 | 71 | 95 | 0.114458 | 0.138554 |
| utility damage | 21 | 0.102 | 0.458214 | 0.476372 | -0.018158 | 11 | 10 | 0.142857 | 0.476190 |
| active smoke/inferno | 156 | 0.757 | 0.357462 | 0.407298 | -0.049837 | 71 | 85 | 0.121795 | 0.147436 |
| recent utility last 5s | 10 | 0.049 | 0.403724 | 0.421036 | -0.017311 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 206 | 1.000 | 0.362066 | 0.422756 | -0.060690 | 73 | 133 | 0.131068 | 0.169903 |

## Active Smoke/Inferno Intervals

- `8.5s` - `46.0s`, rows `76`
- `54.0s` - `60.5s`, rows `14`
- `61.5s` - `68.0s`, rows `14`
- `69.0s` - `94.5s`, rows `52`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `92.0`, LSTM `0.2148`, XGBoost `0.4861`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.2244`, XGBoost `0.4938`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.2187`, XGBoost `0.4861`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.2403`, XGBoost `0.5059`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.2418`, XGBoost `0.4976`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.2428`, XGBoost `0.4976`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.2437`, XGBoost `0.4976`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.2461`, XGBoost `0.4976`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.2501`, XGBoost `0.4976`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.2539`, XGBoost `0.4976`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
