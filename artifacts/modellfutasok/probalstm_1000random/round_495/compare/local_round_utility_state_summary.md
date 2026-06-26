# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `10`
- rows: `234`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 234 | 1.000 | 0.494668 | 0.532561 | -0.037892 | 77 | 157 | 0.487179 | 0.564103 |
| active/recent utility | 234 | 1.000 | 0.494668 | 0.532561 | -0.037892 | 77 | 157 | 0.487179 | 0.564103 |
| strong utility action | 171 | 0.731 | 0.442360 | 0.488408 | -0.046047 | 59 | 112 | 0.356725 | 0.461988 |
| utility damage | 20 | 0.085 | 0.658499 | 0.593768 | 0.064732 | 20 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 160 | 0.684 | 0.432353 | 0.482763 | -0.050409 | 50 | 110 | 0.312500 | 0.425000 |
| recent utility last 5s | 10 | 0.043 | 0.579377 | 0.570577 | 0.008800 | 8 | 2 | 1.000000 | 1.000000 |
| flash effect present | 234 | 1.000 | 0.494668 | 0.532561 | -0.037892 | 77 | 157 | 0.487179 | 0.564103 |

## Active Smoke/Inferno Intervals

- `10.0s` - `33.0s`, rows `47`
- `34.5s` - `56.0s`, rows `44`
- `60.5s` - `94.5s`, rows `69`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.0`, LSTM `0.1652`, XGBoost `0.4139`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `52.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.1727`, XGBoost `0.4187`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.1792`, XGBoost `0.4239`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.1755`, XGBoost `0.4187`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.1734`, XGBoost `0.4160`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.1818`, XGBoost `0.4239`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.1769`, XGBoost `0.4188`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.1788`, XGBoost `0.4160`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.1774`, XGBoost `0.4139`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.1833`, XGBoost `0.4188`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
