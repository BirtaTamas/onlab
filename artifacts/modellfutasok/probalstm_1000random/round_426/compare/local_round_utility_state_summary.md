# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-falcons-vs-3dmax-bo3-XHM3Ovc8L9TfLFTYQFrGdT/falcons-vs-3dmax-m3-dust2.csv`
- round_num: `4`
- rows: `180`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 180 | 1.000 | 0.699306 | 0.799726 | -0.100420 | 0 | 180 | 0.861111 | 1.000000 |
| active/recent utility | 180 | 1.000 | 0.699306 | 0.799726 | -0.100420 | 0 | 180 | 0.861111 | 1.000000 |
| strong utility action | 127 | 0.706 | 0.673334 | 0.782671 | -0.109337 | 0 | 127 | 0.834646 | 1.000000 |
| utility damage | 31 | 0.172 | 0.672943 | 0.779680 | -0.106737 | 0 | 31 | 0.838710 | 1.000000 |
| active smoke/inferno | 125 | 0.694 | 0.677150 | 0.786051 | -0.108901 | 0 | 125 | 0.848000 | 1.000000 |
| recent utility last 5s | 17 | 0.094 | 0.426110 | 0.567099 | -0.140988 | 0 | 17 | 0.000000 | 1.000000 |
| flash effect present | 180 | 1.000 | 0.699306 | 0.799726 | -0.100420 | 0 | 180 | 0.861111 | 1.000000 |

## Active Smoke/Inferno Intervals

- `3.0s` - `43.0s`, rows `81`
- `59.5s` - `81.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `79.5`, LSTM `0.5866`, XGBoost `0.8113`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `20.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.5947`, XGBoost `0.8187`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `20.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.6093`, XGBoost `0.8303`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.6133`, XGBoost `0.8303`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.6141`, XGBoost `0.8303`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.5555`, XGBoost `0.7694`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `1.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.6254`, XGBoost `0.8301`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.5752`, XGBoost `0.7797`, closer `xgboost`, smoke `1`, inferno `4`, utility_damage `34.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.6058`, XGBoost `0.8080`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `20.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.7385`, XGBoost `0.9398`, closer `xgboost`, smoke `1`, inferno `4`, utility_damage `34.0`, recent_utility `0`
