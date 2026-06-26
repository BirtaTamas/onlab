# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `9`
- rows: `228`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 228 | 1.000 | 0.405691 | 0.412739 | -0.007048 | 132 | 96 | 0.364035 | 0.372807 |
| active/recent utility | 228 | 1.000 | 0.405691 | 0.412739 | -0.007048 | 132 | 96 | 0.364035 | 0.372807 |
| strong utility action | 166 | 0.728 | 0.473441 | 0.473069 | 0.000371 | 89 | 77 | 0.246988 | 0.259036 |
| utility damage | 26 | 0.114 | 0.535682 | 0.517853 | 0.017828 | 4 | 22 | 0.153846 | 0.230769 |
| active smoke/inferno | 166 | 0.728 | 0.473441 | 0.473069 | 0.000371 | 89 | 77 | 0.246988 | 0.259036 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 228 | 1.000 | 0.405691 | 0.412739 | -0.007048 | 132 | 96 | 0.364035 | 0.372807 |

## Active Smoke/Inferno Intervals

- `9.5s` - `85.0s`, rows `152`
- `106.5s` - `113.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `108.5`, LSTM `0.6333`, XGBoost `0.8174`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.6446`, XGBoost `0.8174`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.6885`, XGBoost `0.8174`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.0680`, XGBoost `0.1902`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.0916`, XGBoost `0.2097`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `110.0`, LSTM `0.6958`, XGBoost `0.8135`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6406`, XGBoost `0.5240`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6368`, XGBoost `0.5221`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `110.5`, LSTM `0.7020`, XGBoost `0.8135`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6293`, XGBoost `0.5221`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
