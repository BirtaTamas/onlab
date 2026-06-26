# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-gamerlegion-vs-the-mongolz-bo3-bupFip4WbObttNLCPYz_Zo/gamerlegion-vs-the-mongolz-m2-inferno.csv`
- round_num: `16`
- rows: `135`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.530569 | 0.539016 | -0.008446 | 50 | 85 | 0.681481 | 0.911111 |
| active/recent utility | 135 | 1.000 | 0.530569 | 0.539016 | -0.008446 | 50 | 85 | 0.681481 | 0.911111 |
| strong utility action | 128 | 0.948 | 0.529620 | 0.540621 | -0.011002 | 44 | 84 | 0.664062 | 0.906250 |
| utility damage | 20 | 0.148 | 0.453348 | 0.480914 | -0.027566 | 1 | 19 | 0.550000 | 0.550000 |
| active smoke/inferno | 116 | 0.859 | 0.529154 | 0.543504 | -0.014350 | 33 | 83 | 0.629310 | 0.896552 |
| recent utility last 5s | 12 | 0.089 | 0.534123 | 0.512756 | 0.021366 | 11 | 1 | 1.000000 | 1.000000 |
| flash effect present | 135 | 1.000 | 0.530569 | 0.539016 | -0.008446 | 50 | 85 | 0.681481 | 0.911111 |

## Active Smoke/Inferno Intervals

- `9.5s` - `67.0s`, rows `116`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.0`, LSTM `0.6834`, XGBoost `0.7834`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.4187`, XGBoost `0.5173`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.4194`, XGBoost `0.5169`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.5871`, XGBoost `0.6804`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.8339`, XGBoost `0.9206`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.5950`, XGBoost `0.6804`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.5923`, XGBoost `0.5082`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.5903`, XGBoost `0.5063`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.5977`, XGBoost `0.6804`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.1509`, XGBoost `0.2309`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `8.0`, recent_utility `0`
