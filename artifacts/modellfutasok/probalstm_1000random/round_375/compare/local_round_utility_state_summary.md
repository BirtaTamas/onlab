# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-b8-vs-lynn-vision-bo3-Whl3pjYuIoHffY1VOn8vws/b8-vs-lynn-vision-m1-dust2.csv`
- round_num: `11`
- rows: `136`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 136 | 1.000 | 0.173863 | 0.339369 | -0.165506 | 2 | 134 | 0.058824 | 0.058824 |
| active/recent utility | 136 | 1.000 | 0.173863 | 0.339369 | -0.165506 | 2 | 134 | 0.058824 | 0.058824 |
| strong utility action | 92 | 0.676 | 0.098345 | 0.274899 | -0.176554 | 0 | 92 | 0.000000 | 0.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 92 | 0.676 | 0.098345 | 0.274899 | -0.176554 | 0 | 92 | 0.000000 | 0.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 136 | 1.000 | 0.173863 | 0.339369 | -0.165506 | 2 | 134 | 0.058824 | 0.058824 |

## Active Smoke/Inferno Intervals

- `8.5s` - `15.0s`, rows `14`
- `16.5s` - `48.0s`, rows `64`
- `53.5s` - `60.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `60.0`, LSTM `0.1673`, XGBoost `0.4809`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.0300`, XGBoost `0.3102`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.2232`, XGBoost `0.4831`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0496`, XGBoost `0.2705`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0477`, XGBoost `0.2680`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.0519`, XGBoost `0.2712`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0522`, XGBoost `0.2712`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0498`, XGBoost `0.2674`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0545`, XGBoost `0.2712`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.2669`, XGBoost `0.4834`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
