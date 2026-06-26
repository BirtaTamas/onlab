# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `9`
- rows: `239`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 239 | 1.000 | 0.319384 | 0.299360 | 0.020024 | 138 | 101 | 0.569038 | 0.564854 |
| active/recent utility | 239 | 1.000 | 0.319384 | 0.299360 | 0.020024 | 138 | 101 | 0.569038 | 0.564854 |
| strong utility action | 174 | 0.728 | 0.394354 | 0.363339 | 0.031015 | 85 | 89 | 0.477011 | 0.471264 |
| utility damage | 10 | 0.042 | 0.616672 | 0.589066 | 0.027606 | 2 | 8 | 0.000000 | 0.000000 |
| active smoke/inferno | 174 | 0.728 | 0.394354 | 0.363339 | 0.031015 | 85 | 89 | 0.477011 | 0.471264 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 239 | 1.000 | 0.319384 | 0.299360 | 0.020024 | 138 | 101 | 0.569038 | 0.564854 |

## Active Smoke/Inferno Intervals

- `6.0s` - `92.5s`, rows `174`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `20.5`, LSTM `0.5447`, XGBoost `0.2500`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.4915`, XGBoost `0.2185`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5110`, XGBoost `0.2462`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5091`, XGBoost `0.2450`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.4754`, XGBoost `0.2220`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.4808`, XGBoost `0.2318`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.4695`, XGBoost `0.2333`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.4859`, XGBoost `0.2537`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.4693`, XGBoost `0.2537`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.4633`, XGBoost `0.2503`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
