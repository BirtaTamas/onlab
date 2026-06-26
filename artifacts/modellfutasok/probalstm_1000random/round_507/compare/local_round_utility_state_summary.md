# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `7`
- rows: `143`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 143 | 1.000 | 0.035119 | 0.124263 | -0.089145 | 143 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 143 | 1.000 | 0.035119 | 0.124263 | -0.089145 | 143 | 0 | 1.000000 | 1.000000 |
| strong utility action | 83 | 0.580 | 0.047500 | 0.158517 | -0.111017 | 83 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.070 | 0.093526 | 0.222521 | -0.128995 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 83 | 0.580 | 0.047500 | 0.158517 | -0.111017 | 83 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 143 | 1.000 | 0.035119 | 0.124263 | -0.089145 | 143 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `47.5s`, rows `83`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.0`, LSTM `0.0382`, XGBoost `0.2369`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0395`, XGBoost `0.2364`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0411`, XGBoost `0.2369`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0526`, XGBoost `0.2411`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0696`, XGBoost `0.2377`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.0848`, XGBoost `0.2413`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.0324`, XGBoost `0.1823`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0264`, XGBoost `0.1672`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.0254`, XGBoost `0.1629`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.1007`, XGBoost `0.2333`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `14.0`, recent_utility `0`
