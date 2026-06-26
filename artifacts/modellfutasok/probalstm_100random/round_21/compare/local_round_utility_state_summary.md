# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-legacy-bo3-NvI4DRplwm0O-zy6YVkFbj/wildcard-vs-legacy-m2-nuke.csv`
- round_num: `4`
- rows: `140`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 140 | 1.000 | 0.914098 | 0.981531 | -0.067433 | 0 | 140 | 1.000000 | 1.000000 |
| active/recent utility | 140 | 1.000 | 0.914098 | 0.981531 | -0.067433 | 0 | 140 | 1.000000 | 1.000000 |
| strong utility action | 124 | 0.886 | 0.922439 | 0.983391 | -0.060952 | 0 | 124 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 124 | 0.886 | 0.922439 | 0.983391 | -0.060952 | 0 | 124 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 140 | 1.000 | 0.914098 | 0.981531 | -0.067433 | 0 | 140 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `69.0s`, rows `124`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `8.0`, LSTM `0.8335`, XGBoost `0.9666`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.8312`, XGBoost `0.9626`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.8415`, XGBoost `0.9664`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.8530`, XGBoost `0.9664`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.8579`, XGBoost `0.9661`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.8745`, XGBoost `0.9781`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.8647`, XGBoost `0.9666`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.8651`, XGBoost `0.9664`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.8666`, XGBoost `0.9667`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.8665`, XGBoost `0.9664`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
