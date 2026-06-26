# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `29`
- rows: `240`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 240 | 1.000 | 0.712504 | 0.716255 | -0.003751 | 113 | 127 | 0.991667 | 0.975000 |
| active/recent utility | 240 | 1.000 | 0.712504 | 0.716255 | -0.003751 | 113 | 127 | 0.991667 | 0.975000 |
| strong utility action | 187 | 0.779 | 0.668493 | 0.668453 | 0.000040 | 104 | 83 | 0.989305 | 0.967914 |
| utility damage | 30 | 0.125 | 0.693135 | 0.681497 | 0.011638 | 16 | 14 | 0.933333 | 1.000000 |
| active smoke/inferno | 167 | 0.696 | 0.652191 | 0.657405 | -0.005213 | 94 | 73 | 0.988024 | 0.964072 |
| recent utility last 5s | 10 | 0.042 | 0.629870 | 0.526195 | 0.103675 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 240 | 1.000 | 0.712504 | 0.716255 | -0.003751 | 113 | 127 | 0.991667 | 0.975000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `49.0s`, rows `85`
- `52.0s` - `92.5s`, rows `82`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `59.0`, LSTM `0.5877`, XGBoost `0.7204`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `112.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.5946`, XGBoost `0.7204`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `112.0`, recent_utility `0`
- seconds `4.5`, LSTM `0.6563`, XGBoost `0.5346`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.6432`, XGBoost `0.5222`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.6522`, XGBoost `0.5323`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `71.0`, LSTM `0.7791`, XGBoost `0.8988`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `3.5`, LSTM `0.6450`, XGBoost `0.5308`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `66.0`, LSTM `0.7809`, XGBoost `0.8945`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.7858`, XGBoost `0.8989`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.7847`, XGBoost `0.8975`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
