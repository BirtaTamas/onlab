# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `7`
- rows: `184`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 184 | 1.000 | 0.177375 | 0.276561 | -0.099186 | 178 | 6 | 1.000000 | 1.000000 |
| active/recent utility | 184 | 1.000 | 0.177375 | 0.276561 | -0.099186 | 178 | 6 | 1.000000 | 1.000000 |
| strong utility action | 148 | 0.804 | 0.172687 | 0.277284 | -0.104597 | 144 | 4 | 1.000000 | 1.000000 |
| utility damage | 24 | 0.130 | 0.196472 | 0.284314 | -0.087842 | 24 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 147 | 0.799 | 0.171933 | 0.276994 | -0.105061 | 143 | 4 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.054 | 0.115749 | 0.333450 | -0.217701 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 184 | 1.000 | 0.177375 | 0.276561 | -0.099186 | 178 | 6 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `83.5s`, rows `147`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `77.5`, LSTM `0.0413`, XGBoost `0.3083`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `78.0`, LSTM `0.0476`, XGBoost `0.3112`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `77.0`, LSTM `0.0313`, XGBoost `0.2939`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `62.5`, LSTM `0.0301`, XGBoost `0.2742`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.1456`, XGBoost `0.3709`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.0278`, XGBoost `0.2480`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.1057`, XGBoost `0.3241`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `78.5`, LSTM `0.0973`, XGBoost `0.3134`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `59.5`, LSTM `0.0387`, XGBoost `0.2485`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `5.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.1617`, XGBoost `0.3709`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
