# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `18`
- rows: `127`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 127 | 1.000 | 0.825319 | 0.889703 | -0.064385 | 0 | 127 | 1.000000 | 1.000000 |
| active/recent utility | 127 | 1.000 | 0.825319 | 0.889703 | -0.064385 | 0 | 127 | 1.000000 | 1.000000 |
| strong utility action | 106 | 0.835 | 0.839999 | 0.909251 | -0.069252 | 0 | 106 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 106 | 0.835 | 0.839999 | 0.909251 | -0.069252 | 0 | 106 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 127 | 1.000 | 0.825319 | 0.889703 | -0.064385 | 0 | 127 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `61.0s`, rows `106`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.5`, LSTM `0.6136`, XGBoost `0.7436`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.6175`, XGBoost `0.7447`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.8173`, XGBoost `0.9404`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.8212`, XGBoost `0.9404`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.8244`, XGBoost `0.9404`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6331`, XGBoost `0.7448`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.8295`, XGBoost `0.9406`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6393`, XGBoost `0.7445`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.8375`, XGBoost `0.9399`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.7420`, XGBoost `0.8436`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
