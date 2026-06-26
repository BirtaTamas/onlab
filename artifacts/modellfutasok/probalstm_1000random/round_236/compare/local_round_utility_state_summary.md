# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `23`
- rows: `249`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 249 | 1.000 | 0.352945 | 0.457186 | -0.104241 | 245 | 4 | 0.987952 | 0.236948 |
| active/recent utility | 249 | 1.000 | 0.352945 | 0.457186 | -0.104241 | 245 | 4 | 0.987952 | 0.236948 |
| strong utility action | 233 | 0.936 | 0.353779 | 0.452874 | -0.099094 | 229 | 4 | 0.987124 | 0.253219 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 233 | 0.936 | 0.353779 | 0.452874 | -0.099094 | 229 | 4 | 0.987124 | 0.253219 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 249 | 1.000 | 0.352945 | 0.457186 | -0.104241 | 245 | 4 | 0.987952 | 0.236948 |

## Active Smoke/Inferno Intervals

- `8.0s` - `124.0s`, rows `233`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.5`, LSTM `0.3976`, XGBoost `0.6295`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.4076`, XGBoost `0.6295`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.4084`, XGBoost `0.6295`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.4208`, XGBoost `0.6295`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3295`, XGBoost `0.5204`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.4395`, XGBoost `0.6283`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.4407`, XGBoost `0.6290`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.4375`, XGBoost `0.6252`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.4423`, XGBoost `0.6295`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.4411`, XGBoost `0.6283`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
