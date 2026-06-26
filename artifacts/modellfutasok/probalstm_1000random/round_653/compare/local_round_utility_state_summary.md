# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `15`
- rows: `152`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 152 | 1.000 | 0.005375 | 0.021543 | -0.016168 | 81 | 71 | 1.000000 | 1.000000 |
| active/recent utility | 152 | 1.000 | 0.005375 | 0.021543 | -0.016168 | 81 | 71 | 1.000000 | 1.000000 |
| strong utility action | 106 | 0.697 | 0.005446 | 0.019904 | -0.014459 | 66 | 40 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 106 | 0.697 | 0.005446 | 0.019904 | -0.014459 | 66 | 40 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 152 | 1.000 | 0.005375 | 0.021543 | -0.016168 | 81 | 71 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `60.0s`, rows `106`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.0`, LSTM `0.0100`, XGBoost `0.0811`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.0118`, XGBoost `0.0803`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0118`, XGBoost `0.0801`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0135`, XGBoost `0.0801`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0097`, XGBoost `0.0738`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.0102`, XGBoost `0.0707`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0130`, XGBoost `0.0732`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0101`, XGBoost `0.0696`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0093`, XGBoost `0.0667`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0079`, XGBoost `0.0640`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
