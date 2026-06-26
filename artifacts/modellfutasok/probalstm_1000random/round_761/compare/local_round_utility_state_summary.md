# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-tyloo-ancient-6bJQWEKo0L9rHQMGqH72Vs/og-vs-tyloo-ancient.csv`
- round_num: `9`
- rows: `197`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 197 | 1.000 | 0.458392 | 0.433986 | 0.024407 | 62 | 135 | 0.233503 | 0.563452 |
| active/recent utility | 197 | 1.000 | 0.458392 | 0.433986 | 0.024407 | 62 | 135 | 0.233503 | 0.563452 |
| strong utility action | 144 | 0.731 | 0.527367 | 0.502300 | 0.025067 | 36 | 108 | 0.097222 | 0.506944 |
| utility damage | 40 | 0.203 | 0.546651 | 0.509961 | 0.036690 | 2 | 38 | 0.000000 | 0.300000 |
| active smoke/inferno | 144 | 0.731 | 0.527367 | 0.502300 | 0.025067 | 36 | 108 | 0.097222 | 0.506944 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 197 | 1.000 | 0.458392 | 0.433986 | 0.024407 | 62 | 135 | 0.233503 | 0.563452 |

## Active Smoke/Inferno Intervals

- `7.0s` - `71.5s`, rows `130`
- `79.5s` - `86.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.0`, LSTM `0.7927`, XGBoost `0.7035`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5514`, XGBoost `0.4633`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `63.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.5718`, XGBoost `0.4887`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `68.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.5629`, XGBoost `0.4816`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5573`, XGBoost `0.4768`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `62.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.5597`, XGBoost `0.4796`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.5596`, XGBoost `0.4800`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5503`, XGBoost `0.4715`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `38.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.1558`, XGBoost `0.0772`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.5676`, XGBoost `0.4894`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `72.0`, recent_utility `0`
