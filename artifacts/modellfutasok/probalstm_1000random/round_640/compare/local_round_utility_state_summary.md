# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `2`
- rows: `265`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 265 | 1.000 | 0.136110 | 0.194735 | -0.058625 | 229 | 36 | 1.000000 | 1.000000 |
| active/recent utility | 265 | 1.000 | 0.136110 | 0.194735 | -0.058625 | 229 | 36 | 1.000000 | 1.000000 |
| strong utility action | 125 | 0.472 | 0.227569 | 0.286021 | -0.058452 | 89 | 36 | 1.000000 | 1.000000 |
| utility damage | 28 | 0.106 | 0.159512 | 0.308688 | -0.149177 | 28 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 115 | 0.434 | 0.244526 | 0.293292 | -0.048765 | 79 | 36 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 265 | 1.000 | 0.136110 | 0.194735 | -0.058625 | 229 | 36 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `23.5s`, rows `28`
- `39.5s` - `82.5s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `18.5`, LSTM `0.1628`, XGBoost `0.3688`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.1655`, XGBoost `0.3684`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.1691`, XGBoost `0.3684`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.1815`, XGBoost `0.3688`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0287`, XGBoost `0.2081`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `35.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.0294`, XGBoost `0.2059`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `35.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.0261`, XGBoost `0.2022`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `35.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.0288`, XGBoost `0.2001`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `35.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0290`, XGBoost `0.2001`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `35.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0293`, XGBoost `0.2001`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `35.0`, recent_utility `0`
