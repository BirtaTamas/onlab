# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `9`
- rows: `203`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 203 | 1.000 | 0.678327 | 0.656243 | 0.022085 | 152 | 51 | 0.995074 | 0.995074 |
| active/recent utility | 203 | 1.000 | 0.678327 | 0.656243 | 0.022085 | 152 | 51 | 0.995074 | 0.995074 |
| strong utility action | 123 | 0.606 | 0.676824 | 0.666918 | 0.009906 | 78 | 45 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.049 | 0.573435 | 0.568677 | 0.004758 | 6 | 4 | 1.000000 | 1.000000 |
| active smoke/inferno | 120 | 0.591 | 0.678941 | 0.669683 | 0.009258 | 75 | 45 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.049 | 0.582484 | 0.558600 | 0.023884 | 8 | 2 | 1.000000 | 1.000000 |
| flash effect present | 203 | 1.000 | 0.678327 | 0.656243 | 0.022085 | 152 | 51 | 0.995074 | 0.995074 |

## Active Smoke/Inferno Intervals

- `6.5s` - `35.5s`, rows `59`
- `37.0s` - `60.0s`, rows `47`
- `94.5s` - `101.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `98.5`, LSTM `0.5176`, XGBoost `0.6381`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.8627`, XGBoost `0.7749`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.6608`, XGBoost `0.7466`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.6511`, XGBoost `0.7368`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.6520`, XGBoost `0.7362`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.5879`, XGBoost `0.5056`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.6593`, XGBoost `0.7370`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5936`, XGBoost `0.5197`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5908`, XGBoost `0.5188`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5926`, XGBoost `0.5208`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
