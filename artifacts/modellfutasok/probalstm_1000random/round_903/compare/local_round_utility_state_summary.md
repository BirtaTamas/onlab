# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-mouz-vs-falcons-bo3-OIe4ELGS25ekkV8Rf6FbR4/mouz-vs-falcons-m3-mirage.csv`
- round_num: `12`
- rows: `252`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 252 | 1.000 | 0.413823 | 0.457011 | -0.043188 | 184 | 68 | 0.396825 | 0.305556 |
| active/recent utility | 252 | 1.000 | 0.413823 | 0.457011 | -0.043188 | 184 | 68 | 0.396825 | 0.305556 |
| strong utility action | 160 | 0.635 | 0.531770 | 0.549473 | -0.017703 | 100 | 60 | 0.168750 | 0.168750 |
| utility damage | 11 | 0.044 | 0.301976 | 0.272324 | 0.029652 | 4 | 7 | 0.727273 | 0.727273 |
| active smoke/inferno | 159 | 0.631 | 0.534874 | 0.552688 | -0.017813 | 99 | 60 | 0.163522 | 0.163522 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 252 | 1.000 | 0.413823 | 0.457011 | -0.043188 | 184 | 68 | 0.396825 | 0.305556 |

## Active Smoke/Inferno Intervals

- `6.0s` - `85.0s`, rows `159`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.0`, LSTM `0.3323`, XGBoost `0.1700`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.1862`, XGBoost `0.3459`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.2017`, XGBoost `0.3471`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.2046`, XGBoost `0.3459`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.2060`, XGBoost `0.3459`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.1847`, XGBoost `0.3017`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.1984`, XGBoost `0.3071`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.2791`, XGBoost `0.1715`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.5699`, XGBoost `0.6773`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.5141`, XGBoost `0.6198`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
