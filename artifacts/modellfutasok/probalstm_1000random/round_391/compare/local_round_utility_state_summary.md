# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-virtuspro-vs-spirit-bo3-KJqZR5yNeHXaNsc7MGaDWB/virtus-pro-vs-spirit-m1-train.csv`
- round_num: `13`
- rows: `173`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.460993 | 0.511165 | -0.050172 | 162 | 11 | 0.294798 | 0.213873 |
| active/recent utility | 173 | 1.000 | 0.460993 | 0.511165 | -0.050172 | 162 | 11 | 0.294798 | 0.213873 |
| strong utility action | 90 | 0.520 | 0.449783 | 0.505376 | -0.055592 | 79 | 11 | 0.488889 | 0.333333 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 90 | 0.520 | 0.449783 | 0.505376 | -0.055592 | 79 | 11 | 0.488889 | 0.333333 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 173 | 1.000 | 0.460993 | 0.511165 | -0.050172 | 162 | 11 | 0.294798 | 0.213873 |

## Active Smoke/Inferno Intervals

- `38.0s` - `82.5s`, rows `90`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.0`, LSTM `0.5104`, XGBoost `0.7382`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.2770`, XGBoost `0.4869`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.2933`, XGBoost `0.4869`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.2798`, XGBoost `0.1010`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.3210`, XGBoost `0.4947`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.5625`, XGBoost `0.7335`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.5717`, XGBoost `0.7382`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.5709`, XGBoost `0.7236`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.3458`, XGBoost `0.4960`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.5871`, XGBoost `0.7335`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
