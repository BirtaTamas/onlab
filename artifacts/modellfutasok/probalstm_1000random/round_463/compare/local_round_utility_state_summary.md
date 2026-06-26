# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `18`
- rows: `186`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 186 | 1.000 | 0.731885 | 0.755608 | -0.023723 | 84 | 102 | 1.000000 | 1.000000 |
| active/recent utility | 186 | 1.000 | 0.731885 | 0.755608 | -0.023723 | 84 | 102 | 1.000000 | 1.000000 |
| strong utility action | 137 | 0.737 | 0.715720 | 0.731703 | -0.015983 | 74 | 63 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.108 | 0.860853 | 0.872250 | -0.011397 | 9 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 127 | 0.683 | 0.719557 | 0.742642 | -0.023086 | 64 | 63 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.108 | 0.668351 | 0.629838 | 0.038513 | 14 | 6 | 1.000000 | 1.000000 |
| flash effect present | 186 | 1.000 | 0.731885 | 0.755608 | -0.023723 | 84 | 102 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `73.0s`, rows `127`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `68.5`, LSTM `0.6338`, XGBoost `0.8691`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.6465`, XGBoost `0.8687`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.6585`, XGBoost `0.8694`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.6808`, XGBoost `0.8846`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.6662`, XGBoost `0.8687`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.6976`, XGBoost `0.8890`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.7067`, XGBoost `0.8853`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.7045`, XGBoost `0.8831`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.7207`, XGBoost `0.8883`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.7187`, XGBoost `0.8854`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
