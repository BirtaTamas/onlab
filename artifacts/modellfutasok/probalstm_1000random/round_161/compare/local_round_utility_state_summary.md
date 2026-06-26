# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `14`
- rows: `206`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 206 | 1.000 | 0.026742 | 0.037902 | -0.011159 | 161 | 45 | 1.000000 | 1.000000 |
| active/recent utility | 206 | 1.000 | 0.026742 | 0.037902 | -0.011159 | 161 | 45 | 1.000000 | 1.000000 |
| strong utility action | 106 | 0.515 | 0.028121 | 0.038328 | -0.010207 | 68 | 38 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 96 | 0.466 | 0.028612 | 0.038225 | -0.009612 | 58 | 38 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.049 | 0.023405 | 0.039322 | -0.015917 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 206 | 1.000 | 0.026742 | 0.037902 | -0.011159 | 161 | 45 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `30.5s`, rows `44`
- `71.5s` - `97.0s`, rows `52`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `89.5`, LSTM `0.0069`, XGBoost `0.0345`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.0063`, XGBoost `0.0337`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.0072`, XGBoost `0.0346`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.0060`, XGBoost `0.0333`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.0070`, XGBoost `0.0342`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.0070`, XGBoost `0.0342`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.0069`, XGBoost `0.0338`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.0073`, XGBoost `0.0342`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.0071`, XGBoost `0.0338`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.0064`, XGBoost `0.0326`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
