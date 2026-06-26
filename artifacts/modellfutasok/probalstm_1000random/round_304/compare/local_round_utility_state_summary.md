# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-mibr-vs-legacy-nuke-uERfHmzId5aHOSWUmDGvHY/mibr-vs-legacy-nuke.csv`
- round_num: `9`
- rows: `233`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 233 | 1.000 | 0.420434 | 0.400742 | 0.019692 | 88 | 145 | 0.381974 | 0.369099 |
| active/recent utility | 233 | 1.000 | 0.420434 | 0.400742 | 0.019692 | 88 | 145 | 0.381974 | 0.369099 |
| strong utility action | 185 | 0.794 | 0.475412 | 0.449344 | 0.026068 | 46 | 139 | 0.302703 | 0.286486 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 185 | 0.794 | 0.475412 | 0.449344 | 0.026068 | 46 | 139 | 0.302703 | 0.286486 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 233 | 1.000 | 0.420434 | 0.400742 | 0.019692 | 88 | 145 | 0.381974 | 0.369099 |

## Active Smoke/Inferno Intervals

- `7.5s` - `77.5s`, rows `141`
- `79.5s` - `101.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `87.5`, LSTM `0.4215`, XGBoost `0.2105`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.4048`, XGBoost `0.2104`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.4023`, XGBoost `0.2089`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.4018`, XGBoost `0.2103`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.5009`, XGBoost `0.3297`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.4954`, XGBoost `0.3325`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.4947`, XGBoost `0.3325`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.4898`, XGBoost `0.3303`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.4885`, XGBoost `0.3325`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.4754`, XGBoost `0.3204`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
