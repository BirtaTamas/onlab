# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `6`
- rows: `160`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 160 | 1.000 | 0.166996 | 0.199518 | -0.032522 | 142 | 18 | 0.950000 | 1.000000 |
| active/recent utility | 160 | 1.000 | 0.166996 | 0.199518 | -0.032522 | 142 | 18 | 0.950000 | 1.000000 |
| strong utility action | 92 | 0.575 | 0.201262 | 0.239437 | -0.038175 | 75 | 17 | 0.913043 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 92 | 0.575 | 0.201262 | 0.239437 | -0.038175 | 75 | 17 | 0.913043 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 160 | 1.000 | 0.166996 | 0.199518 | -0.032522 | 142 | 18 | 0.950000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `55.0s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.5`, LSTM `0.1648`, XGBoost `0.3759`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0111`, XGBoost `0.1301`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.0123`, XGBoost `0.1276`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.0352`, XGBoost `0.1486`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.0122`, XGBoost `0.1251`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.0158`, XGBoost `0.1285`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0524`, XGBoost `0.1636`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0122`, XGBoost `0.1233`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.0588`, XGBoost `0.1637`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0662`, XGBoost `0.1685`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
