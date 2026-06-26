# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `3`
- rows: `247`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 247 | 1.000 | 0.090533 | 0.156758 | -0.066225 | 247 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 247 | 1.000 | 0.090533 | 0.156758 | -0.066225 | 247 | 0 | 1.000000 | 1.000000 |
| strong utility action | 171 | 0.692 | 0.117100 | 0.198726 | -0.081627 | 171 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 171 | 0.692 | 0.117100 | 0.198726 | -0.081627 | 171 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 247 | 1.000 | 0.090533 | 0.156758 | -0.066225 | 247 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `69.5s`, rows `125`
- `70.5s` - `93.0s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `73.5`, LSTM `0.0356`, XGBoost `0.2067`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.0404`, XGBoost `0.2079`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.0426`, XGBoost `0.2079`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.0437`, XGBoost `0.2080`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.0441`, XGBoost `0.2079`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.0449`, XGBoost `0.2075`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.0460`, XGBoost `0.2067`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.1304`, XGBoost `0.2735`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.1310`, XGBoost `0.2699`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.1391`, XGBoost `0.2754`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
