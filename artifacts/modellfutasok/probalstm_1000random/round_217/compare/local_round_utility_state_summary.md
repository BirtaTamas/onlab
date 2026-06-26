# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `11`
- rows: `164`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 164 | 1.000 | 0.072128 | 0.112826 | -0.040697 | 164 | 0 | 0.993902 | 0.908537 |
| active/recent utility | 164 | 1.000 | 0.072128 | 0.112826 | -0.040697 | 164 | 0 | 0.993902 | 0.908537 |
| strong utility action | 155 | 0.945 | 0.066741 | 0.106579 | -0.039838 | 155 | 0 | 0.993548 | 0.922581 |
| utility damage | 16 | 0.098 | 0.157389 | 0.208867 | -0.051478 | 16 | 0 | 1.000000 | 0.812500 |
| active smoke/inferno | 145 | 0.884 | 0.038612 | 0.079259 | -0.040647 | 145 | 0 | 0.993103 | 0.951724 |
| recent utility last 5s | 10 | 0.061 | 0.474615 | 0.502722 | -0.028107 | 10 | 0 | 1.000000 | 0.500000 |
| flash effect present | 164 | 1.000 | 0.072128 | 0.112826 | -0.040697 | 164 | 0 | 0.993902 | 0.908537 |

## Active Smoke/Inferno Intervals

- `6.5s` - `78.5s`, rows `145`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `18.5`, LSTM `0.0155`, XGBoost `0.1253`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.0174`, XGBoost `0.1267`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0167`, XGBoost `0.1253`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.0174`, XGBoost `0.1251`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0271`, XGBoost `0.1330`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0259`, XGBoost `0.1314`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.0236`, XGBoost `0.1288`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.0264`, XGBoost `0.1310`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.0270`, XGBoost `0.1314`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.0203`, XGBoost `0.1235`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
