# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m3-inferno.csv`
- round_num: `15`
- rows: `135`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.743026 | 0.703791 | 0.039235 | 90 | 45 | 1.000000 | 1.000000 |
| active/recent utility | 135 | 1.000 | 0.743026 | 0.703791 | 0.039235 | 90 | 45 | 1.000000 | 1.000000 |
| strong utility action | 99 | 0.733 | 0.741600 | 0.718629 | 0.022971 | 64 | 35 | 1.000000 | 1.000000 |
| utility damage | 23 | 0.170 | 0.742066 | 0.687678 | 0.054388 | 23 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 99 | 0.733 | 0.741600 | 0.718629 | 0.022971 | 64 | 35 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 135 | 1.000 | 0.743026 | 0.703791 | 0.039235 | 90 | 45 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `17.0s`, rows `15`
- `20.5s` - `49.5s`, rows `59`
- `55.0s` - `67.0s`, rows `25`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.7969`, XGBoost `0.6256`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.7832`, XGBoost `0.6193`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.7919`, XGBoost `0.6301`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.5983`, XGBoost `0.7562`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.7906`, XGBoost `0.6334`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.7865`, XGBoost `0.6318`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.6026`, XGBoost `0.7562`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.7714`, XGBoost `0.6256`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.6093`, XGBoost `0.7527`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.6125`, XGBoost `0.7527`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
