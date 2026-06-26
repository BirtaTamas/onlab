# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m2-ancient.csv`
- round_num: `10`
- rows: `136`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 136 | 1.000 | 0.261616 | 0.213030 | 0.048586 | 49 | 87 | 0.713235 | 0.720588 |
| active/recent utility | 136 | 1.000 | 0.261616 | 0.213030 | 0.048586 | 49 | 87 | 0.713235 | 0.720588 |
| strong utility action | 109 | 0.801 | 0.239223 | 0.190608 | 0.048614 | 49 | 60 | 0.752294 | 0.761468 |
| utility damage | 17 | 0.125 | 0.556912 | 0.553228 | 0.003684 | 14 | 3 | 0.058824 | 0.117647 |
| active smoke/inferno | 109 | 0.801 | 0.239223 | 0.190608 | 0.048614 | 49 | 60 | 0.752294 | 0.761468 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 136 | 1.000 | 0.261616 | 0.213030 | 0.048586 | 49 | 87 | 0.713235 | 0.720588 |

## Active Smoke/Inferno Intervals

- `6.0s` - `40.5s`, rows `70`
- `48.5s` - `67.5s`, rows `39`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.0`, LSTM `0.3041`, XGBoost `0.0976`, closer `xgboost`, smoke `6`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.2813`, XGBoost `0.0974`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.2735`, XGBoost `0.0974`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.2675`, XGBoost `0.0976`, closer `xgboost`, smoke `6`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.2639`, XGBoost `0.0974`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.2612`, XGBoost `0.0976`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.2597`, XGBoost `0.0976`, closer `xgboost`, smoke `6`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.2621`, XGBoost `0.1007`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.5244`, XGBoost `0.3659`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.2558`, XGBoost `0.0974`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
