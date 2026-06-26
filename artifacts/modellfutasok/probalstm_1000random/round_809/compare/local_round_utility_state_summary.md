# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `21`
- rows: `250`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 250 | 1.000 | 0.246440 | 0.283332 | -0.036892 | 239 | 11 | 0.984000 | 0.740000 |
| active/recent utility | 250 | 1.000 | 0.246440 | 0.283332 | -0.036892 | 239 | 11 | 0.984000 | 0.740000 |
| strong utility action | 158 | 0.632 | 0.335427 | 0.384065 | -0.048638 | 147 | 11 | 0.981013 | 0.689873 |
| utility damage | 32 | 0.128 | 0.191928 | 0.220456 | -0.028528 | 32 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 158 | 0.632 | 0.335427 | 0.384065 | -0.048638 | 147 | 11 | 0.981013 | 0.689873 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 250 | 1.000 | 0.246440 | 0.283332 | -0.036892 | 239 | 11 | 0.984000 | 0.740000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `87.5s`, rows `158`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `72.0`, LSTM `0.0702`, XGBoost `0.2142`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.0747`, XGBoost `0.2161`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.3695`, XGBoost `0.5109`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.0815`, XGBoost `0.2208`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.3748`, XGBoost `0.5107`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.3753`, XGBoost `0.5109`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.3757`, XGBoost `0.5107`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.3449`, XGBoost `0.4787`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.0540`, XGBoost `0.1868`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.0554`, XGBoost `0.1868`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
