# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-b8-bo3-rUWlZLFFckLiQv1C1wSlHb/g2-vs-b8-m3-ancient.csv`
- round_num: `6`
- rows: `217`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 217 | 1.000 | 0.691653 | 0.645172 | 0.046481 | 167 | 50 | 1.000000 | 1.000000 |
| active/recent utility | 217 | 1.000 | 0.691653 | 0.645172 | 0.046481 | 167 | 50 | 1.000000 | 1.000000 |
| strong utility action | 184 | 0.848 | 0.691167 | 0.639924 | 0.051243 | 143 | 41 | 1.000000 | 1.000000 |
| utility damage | 31 | 0.143 | 0.612422 | 0.591795 | 0.020627 | 25 | 6 | 1.000000 | 1.000000 |
| active smoke/inferno | 184 | 0.848 | 0.691167 | 0.639924 | 0.051243 | 143 | 41 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 217 | 1.000 | 0.691653 | 0.645172 | 0.046481 | 167 | 50 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `50.0s`, rows `88`
- `54.5s` - `79.5s`, rows `51`
- `82.0s` - `104.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.5`, LSTM `0.9006`, XGBoost `0.7484`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.8918`, XGBoost `0.7400`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.8987`, XGBoost `0.7484`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.8953`, XGBoost `0.7458`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.8895`, XGBoost `0.7402`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.8962`, XGBoost `0.7488`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.8952`, XGBoost `0.7488`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.7564`, XGBoost `0.6104`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.8944`, XGBoost `0.7484`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.8947`, XGBoost `0.7488`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
