# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m1-mirage.csv`
- round_num: `8`
- rows: `244`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 244 | 1.000 | 0.585321 | 0.558097 | 0.027224 | 206 | 38 | 0.901639 | 0.918033 |
| active/recent utility | 244 | 1.000 | 0.585321 | 0.558097 | 0.027224 | 206 | 38 | 0.901639 | 0.918033 |
| strong utility action | 212 | 0.869 | 0.585342 | 0.536811 | 0.048531 | 197 | 15 | 0.933962 | 0.933962 |
| utility damage | 20 | 0.082 | 0.623210 | 0.562271 | 0.060938 | 20 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 198 | 0.811 | 0.582698 | 0.535903 | 0.046795 | 183 | 15 | 0.929293 | 0.929293 |
| recent utility last 5s | 22 | 0.090 | 0.626869 | 0.548935 | 0.077934 | 22 | 0 | 1.000000 | 1.000000 |
| flash effect present | 244 | 1.000 | 0.585321 | 0.558097 | 0.027224 | 206 | 38 | 0.901639 | 0.918033 |

## Active Smoke/Inferno Intervals

- `7.0s` - `81.5s`, rows `150`
- `86.5s` - `110.0s`, rows `48`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `103.5`, LSTM `0.4023`, XGBoost `0.2393`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.6541`, XGBoost `0.5077`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.6479`, XGBoost `0.5077`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `89.0`, LSTM `0.7207`, XGBoost `0.5827`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.7196`, XGBoost `0.5837`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.7170`, XGBoost `0.5827`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `110.0`, LSTM `0.0541`, XGBoost `0.1857`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.7067`, XGBoost `0.5754`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.7120`, XGBoost `0.5832`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.7111`, XGBoost `0.5832`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
