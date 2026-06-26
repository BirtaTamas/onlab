# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-heroic-vs-3dmax-bo3-OVT4ch_FfOW2E26liKqT_k/heroic-vs-3dmax-m2-inferno.csv`
- round_num: `17`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.788642 | 0.773010 | 0.015632 | 124 | 106 | 0.843478 | 0.921739 |
| active/recent utility | 230 | 1.000 | 0.788642 | 0.773010 | 0.015632 | 124 | 106 | 0.843478 | 0.921739 |
| strong utility action | 111 | 0.483 | 0.707554 | 0.739457 | -0.031903 | 23 | 88 | 0.837838 | 0.954955 |
| utility damage | 10 | 0.043 | 0.519329 | 0.545849 | -0.026520 | 0 | 10 | 0.800000 | 1.000000 |
| active smoke/inferno | 111 | 0.483 | 0.707554 | 0.739457 | -0.031903 | 23 | 88 | 0.837838 | 0.954955 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.788642 | 0.773010 | 0.015632 | 124 | 106 | 0.843478 | 0.921739 |

## Active Smoke/Inferno Intervals

- `9.0s` - `58.5s`, rows `100`
- `94.0s` - `99.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `39.5`, LSTM `0.7296`, XGBoost `0.8603`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.7415`, XGBoost `0.8597`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.7442`, XGBoost `0.8591`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.7443`, XGBoost `0.8579`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.7582`, XGBoost `0.8707`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `67.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3796`, XGBoost `0.4915`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.7487`, XGBoost `0.8583`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.7515`, XGBoost `0.8579`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.7782`, XGBoost `0.8843`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `67.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.7554`, XGBoost `0.8579`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
