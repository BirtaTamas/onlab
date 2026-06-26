# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `2`
- rows: `178`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 178 | 1.000 | 0.012085 | 0.035758 | -0.023673 | 159 | 19 | 1.000000 | 1.000000 |
| active/recent utility | 178 | 1.000 | 0.012085 | 0.035758 | -0.023673 | 159 | 19 | 1.000000 | 1.000000 |
| strong utility action | 107 | 0.601 | 0.016509 | 0.044372 | -0.027863 | 107 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 107 | 0.601 | 0.016509 | 0.044372 | -0.027863 | 107 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 178 | 1.000 | 0.012085 | 0.035758 | -0.023673 | 159 | 19 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `32.5s`, rows `52`
- `34.5s` - `61.5s`, rows `55`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `61.5`, LSTM `0.0050`, XGBoost `0.0546`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.0042`, XGBoost `0.0520`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0187`, XGBoost `0.0648`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.0206`, XGBoost `0.0665`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.0183`, XGBoost `0.0642`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.0190`, XGBoost `0.0648`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.0190`, XGBoost `0.0642`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0201`, XGBoost `0.0642`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0207`, XGBoost `0.0648`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0209`, XGBoost `0.0648`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
