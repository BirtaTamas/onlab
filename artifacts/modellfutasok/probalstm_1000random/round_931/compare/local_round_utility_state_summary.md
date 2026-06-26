# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `4`
- rows: `134`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 134 | 1.000 | 0.546952 | 0.538734 | 0.008218 | 32 | 102 | 0.208955 | 0.171642 |
| active/recent utility | 134 | 1.000 | 0.546952 | 0.538734 | 0.008218 | 32 | 102 | 0.208955 | 0.171642 |
| strong utility action | 121 | 0.903 | 0.539894 | 0.536111 | 0.003783 | 32 | 89 | 0.231405 | 0.190083 |
| utility damage | 29 | 0.216 | 0.548469 | 0.594580 | -0.046111 | 13 | 16 | 0.172414 | 0.034483 |
| active smoke/inferno | 121 | 0.903 | 0.539894 | 0.536111 | 0.003783 | 32 | 89 | 0.231405 | 0.190083 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 134 | 1.000 | 0.546952 | 0.538734 | 0.008218 | 32 | 102 | 0.208955 | 0.171642 |

## Active Smoke/Inferno Intervals

- `6.5s` - `66.5s`, rows `121`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `66.5`, LSTM `0.1694`, XGBoost `0.5029`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `7.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.2024`, XGBoost `0.5025`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.2154`, XGBoost `0.5025`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.5593`, XGBoost `0.8325`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.5829`, XGBoost `0.8341`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.6047`, XGBoost `0.8333`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.1127`, XGBoost `0.3307`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.1188`, XGBoost `0.3286`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.1713`, XGBoost `0.3756`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.1301`, XGBoost `0.3332`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
