# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-vitality-bo3-ZpOL0o26IrRvvgFRbFxVou/lynn-vision-vs-vitality-m1-dust2.csv`
- round_num: `3`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.604227 | 0.591726 | 0.012500 | 146 | 84 | 0.630435 | 0.286957 |
| active/recent utility | 230 | 1.000 | 0.604227 | 0.591726 | 0.012500 | 146 | 84 | 0.630435 | 0.286957 |
| strong utility action | 192 | 0.835 | 0.604579 | 0.595087 | 0.009492 | 119 | 73 | 0.609375 | 0.296875 |
| utility damage | 10 | 0.043 | 0.502740 | 0.475589 | 0.027151 | 10 | 0 | 0.700000 | 0.000000 |
| active smoke/inferno | 185 | 0.804 | 0.609309 | 0.599755 | 0.009554 | 114 | 71 | 0.627027 | 0.308108 |
| recent utility last 5s | 10 | 0.043 | 0.483166 | 0.472559 | 0.010607 | 8 | 2 | 0.200000 | 0.000000 |
| flash effect present | 230 | 1.000 | 0.604227 | 0.591726 | 0.012500 | 146 | 84 | 0.630435 | 0.286957 |

## Active Smoke/Inferno Intervals

- `9.0s` - `33.5s`, rows `50`
- `43.0s` - `110.0s`, rows `135`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `90.0`, LSTM `0.5917`, XGBoost `0.7463`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.6103`, XGBoost `0.7463`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.6186`, XGBoost `0.7463`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.6217`, XGBoost `0.7249`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5511`, XGBoost `0.4733`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.5477`, XGBoost `0.4730`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.5465`, XGBoost `0.4721`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5468`, XGBoost `0.4726`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.8380`, XGBoost `0.9107`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.5263`, XGBoost `0.4538`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
