# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `10`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.701208 | 0.740891 | -0.039683 | 35 | 195 | 0.960870 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.701208 | 0.740891 | -0.039683 | 35 | 195 | 0.960870 | 1.000000 |
| strong utility action | 212 | 0.922 | 0.718055 | 0.760711 | -0.042656 | 29 | 183 | 1.000000 | 1.000000 |
| utility damage | 19 | 0.083 | 0.748920 | 0.810765 | -0.061845 | 0 | 19 | 1.000000 | 1.000000 |
| active smoke/inferno | 212 | 0.922 | 0.718055 | 0.760711 | -0.042656 | 29 | 183 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.701208 | 0.740891 | -0.039683 | 35 | 195 | 0.960870 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `114.5s`, rows `212`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `87.0`, LSTM `0.7403`, XGBoost `0.8784`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5937`, XGBoost `0.7241`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.5820`, XGBoost `0.7112`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5823`, XGBoost `0.7112`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5963`, XGBoost `0.7239`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.5971`, XGBoost `0.7239`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `114.5`, LSTM `0.8116`, XGBoost `0.9356`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6022`, XGBoost `0.7239`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.7622`, XGBoost `0.6405`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.6035`, XGBoost `0.7239`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
