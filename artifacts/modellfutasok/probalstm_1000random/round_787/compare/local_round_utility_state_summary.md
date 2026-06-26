# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv`
- round_num: `17`
- rows: `161`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 161 | 1.000 | 0.181234 | 0.186116 | -0.004882 | 105 | 56 | 0.863354 | 0.975155 |
| active/recent utility | 161 | 1.000 | 0.181234 | 0.186116 | -0.004882 | 105 | 56 | 0.863354 | 0.975155 |
| strong utility action | 136 | 0.845 | 0.186227 | 0.191853 | -0.005626 | 86 | 50 | 0.882353 | 0.985294 |
| utility damage | 10 | 0.062 | 0.462135 | 0.418586 | 0.043550 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 125 | 0.776 | 0.151513 | 0.165198 | -0.013685 | 86 | 39 | 0.960000 | 0.992000 |
| recent utility last 5s | 11 | 0.068 | 0.580705 | 0.494752 | 0.085953 | 0 | 11 | 0.000000 | 0.909091 |
| flash effect present | 161 | 1.000 | 0.181234 | 0.186116 | -0.004882 | 105 | 56 | 0.863354 | 0.975155 |

## Active Smoke/Inferno Intervals

- `8.5s` - `63.5s`, rows `111`
- `72.5s` - `79.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.0`, LSTM `0.6528`, XGBoost `0.5267`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `1.5`, LSTM `0.5977`, XGBoost `0.4868`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `6.0`, LSTM `0.6028`, XGBoost `0.4923`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.5917`, XGBoost `0.4875`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `35.0`, LSTM `0.0323`, XGBoost `0.1245`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0329`, XGBoost `0.1245`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0338`, XGBoost `0.1242`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0418`, XGBoost `0.1303`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.0379`, XGBoost `0.1252`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.5`, LSTM `0.5780`, XGBoost `0.4909`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
