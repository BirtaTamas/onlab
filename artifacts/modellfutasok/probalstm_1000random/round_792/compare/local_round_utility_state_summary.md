# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-imperial-vs-liquid-bo3-eiIGPV5tjvJFQ73hC8D8JI/imperial-vs-liquid-m3-anubis.csv`
- round_num: `10`
- rows: `217`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 217 | 1.000 | 0.947302 | 0.988361 | -0.041060 | 0 | 217 | 1.000000 | 1.000000 |
| active/recent utility | 217 | 1.000 | 0.947302 | 0.988361 | -0.041060 | 0 | 217 | 1.000000 | 1.000000 |
| strong utility action | 69 | 0.318 | 0.906332 | 0.976311 | -0.069978 | 0 | 69 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 69 | 0.318 | 0.906332 | 0.976311 | -0.069978 | 0 | 69 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 217 | 1.000 | 0.947302 | 0.988361 | -0.041060 | 0 | 217 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `41.5s`, rows `69`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.5`, LSTM `0.7903`, XGBoost `0.9569`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.7991`, XGBoost `0.9569`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.8042`, XGBoost `0.9569`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.8082`, XGBoost `0.9579`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.8115`, XGBoost `0.9569`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.8120`, XGBoost `0.9570`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.8147`, XGBoost `0.9567`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.8200`, XGBoost `0.9580`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.8192`, XGBoost `0.9569`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.8211`, XGBoost `0.9568`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
