# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-b8-vs-hotu-bo3-tmCfOETKzYqjV6vSvNp3-F/b8-vs-hotu-m3-ancient.csv`
- round_num: `6`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.915446 | 0.933895 | -0.018449 | 28 | 202 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.915446 | 0.933895 | -0.018449 | 28 | 202 | 1.000000 | 1.000000 |
| strong utility action | 127 | 0.552 | 0.886365 | 0.921567 | -0.035202 | 16 | 111 | 1.000000 | 1.000000 |
| utility damage | 24 | 0.104 | 0.879020 | 0.883520 | -0.004499 | 9 | 15 | 1.000000 | 1.000000 |
| active smoke/inferno | 127 | 0.552 | 0.886365 | 0.921567 | -0.035202 | 16 | 111 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.915446 | 0.933895 | -0.018449 | 28 | 202 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `69.0s`, rows `127`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.5`, LSTM `0.8039`, XGBoost `0.9334`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.8036`, XGBoost `0.9330`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.8123`, XGBoost `0.9336`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.8130`, XGBoost `0.9333`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.8156`, XGBoost `0.9336`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.8282`, XGBoost `0.9337`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.8280`, XGBoost `0.9330`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.8325`, XGBoost `0.9330`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.7120`, XGBoost `0.6129`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.8393`, XGBoost `0.9379`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
