# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `14`
- rows: `133`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 133 | 1.000 | 0.038598 | 0.052253 | -0.013655 | 111 | 22 | 1.000000 | 1.000000 |
| active/recent utility | 133 | 1.000 | 0.038598 | 0.052253 | -0.013655 | 111 | 22 | 1.000000 | 1.000000 |
| strong utility action | 124 | 0.932 | 0.040604 | 0.051885 | -0.011282 | 102 | 22 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 118 | 0.887 | 0.041457 | 0.051084 | -0.009627 | 96 | 22 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.075 | 0.035470 | 0.068827 | -0.033357 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 133 | 1.000 | 0.038598 | 0.052253 | -0.013655 | 111 | 22 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `66.0s`, rows `118`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `59.5`, LSTM `0.0050`, XGBoost `0.0593`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `4.5`, LSTM `0.0127`, XGBoost `0.0651`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `60.0`, LSTM `0.0076`, XGBoost `0.0598`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.0167`, XGBoost `0.0677`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.0221`, XGBoost `0.0677`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `60.5`, LSTM `0.0071`, XGBoost `0.0524`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.0041`, XGBoost `0.0473`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.0042`, XGBoost `0.0471`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.0050`, XGBoost `0.0473`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.0276`, XGBoost `0.0695`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
