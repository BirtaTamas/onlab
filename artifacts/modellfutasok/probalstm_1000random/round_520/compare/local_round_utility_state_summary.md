# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-gentle-mates-vs-aurora-bo3-gDH2lDrlT5ROvKI-0e6nmI/gentle-mates-vs-aurora-m1-nuke.csv`
- round_num: `5`
- rows: `143`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 143 | 1.000 | 0.672545 | 0.658180 | 0.014364 | 94 | 49 | 1.000000 | 1.000000 |
| active/recent utility | 143 | 1.000 | 0.672545 | 0.658180 | 0.014364 | 94 | 49 | 1.000000 | 1.000000 |
| strong utility action | 117 | 0.818 | 0.649870 | 0.637319 | 0.012551 | 80 | 37 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.140 | 0.633365 | 0.596326 | 0.037040 | 16 | 4 | 1.000000 | 1.000000 |
| active smoke/inferno | 117 | 0.818 | 0.649870 | 0.637319 | 0.012551 | 80 | 37 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 143 | 1.000 | 0.672545 | 0.658180 | 0.014364 | 94 | 49 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `65.0s`, rows `117`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `55.0`, LSTM `0.7441`, XGBoost `0.8875`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.5891`, XGBoost `0.7167`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.5913`, XGBoost `0.7167`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.7637`, XGBoost `0.8875`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.7693`, XGBoost `0.8875`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.5986`, XGBoost `0.7167`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.6131`, XGBoost `0.7167`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.7833`, XGBoost `0.8837`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.6166`, XGBoost `0.7167`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.6367`, XGBoost `0.5393`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
