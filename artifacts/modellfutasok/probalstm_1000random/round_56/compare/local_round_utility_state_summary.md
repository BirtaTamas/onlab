# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `12`
- rows: `114`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 114 | 1.000 | 0.810221 | 0.777605 | 0.032616 | 82 | 32 | 1.000000 | 1.000000 |
| active/recent utility | 114 | 1.000 | 0.810221 | 0.777605 | 0.032616 | 82 | 32 | 1.000000 | 1.000000 |
| strong utility action | 88 | 0.772 | 0.808158 | 0.780876 | 0.027282 | 65 | 23 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.096 | 0.794604 | 0.746397 | 0.048206 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 87 | 0.763 | 0.810000 | 0.783787 | 0.026213 | 64 | 23 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.088 | 0.634232 | 0.531418 | 0.102814 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 114 | 1.000 | 0.810221 | 0.777605 | 0.032616 | 82 | 32 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `49.5s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `6.5`, LSTM `0.6426`, XGBoost `0.5217`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `6.0`, LSTM `0.6479`, XGBoost `0.5276`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.6506`, XGBoost `0.5323`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `9.0`, LSTM `0.6368`, XGBoost `0.5335`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `10.5`, LSTM `0.6348`, XGBoost `0.5332`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `10.0`, LSTM `0.6337`, XGBoost `0.5335`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `11.0`, LSTM `0.6295`, XGBoost `0.5332`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.6281`, XGBoost `0.5327`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `8.5`, LSTM `0.6280`, XGBoost `0.5335`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `13.0`, LSTM `0.6729`, XGBoost `0.7670`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
