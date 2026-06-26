# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `6`
- rows: `204`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 204 | 1.000 | 0.192586 | 0.239387 | -0.046801 | 188 | 16 | 1.000000 | 0.995098 |
| active/recent utility | 204 | 1.000 | 0.192586 | 0.239387 | -0.046801 | 188 | 16 | 1.000000 | 0.995098 |
| strong utility action | 164 | 0.804 | 0.217303 | 0.273938 | -0.056635 | 153 | 11 | 1.000000 | 0.993902 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 153 | 0.750 | 0.198261 | 0.260634 | -0.062373 | 153 | 0 | 1.000000 | 0.993464 |
| recent utility last 5s | 11 | 0.054 | 0.482158 | 0.458987 | 0.023172 | 0 | 11 | 1.000000 | 1.000000 |
| flash effect present | 204 | 1.000 | 0.192586 | 0.239387 | -0.046801 | 188 | 16 | 1.000000 | 0.995098 |

## Active Smoke/Inferno Intervals

- `9.5s` - `85.5s`, rows `153`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.5`, LSTM `0.1864`, XGBoost `0.3767`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.2739`, XGBoost `0.4628`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.2025`, XGBoost `0.3909`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.2768`, XGBoost `0.4628`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.2101`, XGBoost `0.3925`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.2832`, XGBoost `0.4628`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.2773`, XGBoost `0.4563`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.2802`, XGBoost `0.4563`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.2934`, XGBoost `0.4626`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.2978`, XGBoost `0.4626`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
