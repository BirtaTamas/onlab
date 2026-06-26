# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m3-inferno.csv`
- round_num: `8`
- rows: `301`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 301 | 1.000 | 0.176440 | 0.191807 | -0.015367 | 183 | 118 | 0.970100 | 0.930233 |
| active/recent utility | 301 | 1.000 | 0.176440 | 0.191807 | -0.015367 | 183 | 118 | 0.970100 | 0.930233 |
| strong utility action | 255 | 0.847 | 0.169298 | 0.185779 | -0.016481 | 162 | 93 | 0.980392 | 0.996078 |
| utility damage | 10 | 0.033 | 0.349897 | 0.301365 | 0.048532 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 255 | 0.847 | 0.169298 | 0.185779 | -0.016481 | 162 | 93 | 0.980392 | 0.996078 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 301 | 1.000 | 0.176440 | 0.191807 | -0.015367 | 183 | 118 | 0.970100 | 0.930233 |

## Active Smoke/Inferno Intervals

- `10.0s` - `130.5s`, rows `242`
- `144.0s` - `150.0s`, rows `13`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `98.5`, LSTM `0.0646`, XGBoost `0.2358`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.0672`, XGBoost `0.2267`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.0711`, XGBoost `0.2267`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.0360`, XGBoost `0.1911`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.0337`, XGBoost `0.1868`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.0394`, XGBoost `0.1911`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.0378`, XGBoost `0.1865`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.0466`, XGBoost `0.1911`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.0928`, XGBoost `0.2361`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.0479`, XGBoost `0.1818`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
