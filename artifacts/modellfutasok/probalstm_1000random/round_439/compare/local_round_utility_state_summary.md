# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `7`
- rows: `205`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 205 | 1.000 | 0.650232 | 0.668428 | -0.018197 | 80 | 125 | 0.195122 | 0.136585 |
| active/recent utility | 205 | 1.000 | 0.650232 | 0.668428 | -0.018197 | 80 | 125 | 0.195122 | 0.136585 |
| strong utility action | 165 | 0.805 | 0.725959 | 0.717985 | 0.007975 | 49 | 116 | 0.090909 | 0.018182 |
| utility damage | 20 | 0.098 | 0.796050 | 0.742981 | 0.053069 | 7 | 13 | 0.000000 | 0.000000 |
| active smoke/inferno | 165 | 0.805 | 0.725959 | 0.717985 | 0.007975 | 49 | 116 | 0.090909 | 0.018182 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 205 | 1.000 | 0.650232 | 0.668428 | -0.018197 | 80 | 125 | 0.195122 | 0.136585 |

## Active Smoke/Inferno Intervals

- `7.5s` - `89.5s`, rows `165`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `87.5`, LSTM `0.1777`, XGBoost `0.5165`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.1850`, XGBoost `0.5191`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.1802`, XGBoost `0.5112`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.2135`, XGBoost `0.5191`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.2323`, XGBoost `0.5177`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.2978`, XGBoost `0.5157`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.1270`, XGBoost `0.3126`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.1240`, XGBoost `0.3069`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.1384`, XGBoost `0.3135`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.3957`, XGBoost `0.5687`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
