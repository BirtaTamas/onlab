# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `14`
- rows: `157`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.322157 | 0.403617 | -0.081460 | 18 | 139 | 0.197452 | 0.197452 |
| active/recent utility | 157 | 1.000 | 0.322157 | 0.403617 | -0.081460 | 18 | 139 | 0.197452 | 0.197452 |
| strong utility action | 148 | 0.943 | 0.327155 | 0.410752 | -0.083597 | 16 | 132 | 0.209459 | 0.209459 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 138 | 0.879 | 0.325860 | 0.418435 | -0.092575 | 7 | 131 | 0.224638 | 0.224638 |
| recent utility last 5s | 10 | 0.064 | 0.345024 | 0.304727 | 0.040297 | 9 | 1 | 0.000000 | 0.000000 |
| flash effect present | 157 | 1.000 | 0.322157 | 0.403617 | -0.081460 | 18 | 139 | 0.197452 | 0.197452 |

## Active Smoke/Inferno Intervals

- `9.5s` - `78.0s`, rows `138`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.5`, LSTM `0.5779`, XGBoost `0.8161`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.5751`, XGBoost `0.8108`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.5870`, XGBoost `0.8161`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.6146`, XGBoost `0.8334`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.5963`, XGBoost `0.8151`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.6055`, XGBoost `0.8124`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.5520`, XGBoost `0.7562`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.6325`, XGBoost `0.8334`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5568`, XGBoost `0.7575`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.5651`, XGBoost `0.7642`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
