# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-heroic-bo3-ReZhZ3UThZvWjRyUeuYiIR/falcons-vs-heroic-m3-dust2.csv`
- round_num: `8`
- rows: `257`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 257 | 1.000 | 0.353393 | 0.342716 | 0.010677 | 123 | 134 | 0.428016 | 0.420233 |
| active/recent utility | 257 | 1.000 | 0.353393 | 0.342716 | 0.010677 | 123 | 134 | 0.428016 | 0.420233 |
| strong utility action | 195 | 0.759 | 0.405223 | 0.395970 | 0.009252 | 80 | 115 | 0.343590 | 0.333333 |
| utility damage | 20 | 0.078 | 0.604924 | 0.582877 | 0.022047 | 4 | 16 | 0.000000 | 0.000000 |
| active smoke/inferno | 195 | 0.759 | 0.405223 | 0.395970 | 0.009252 | 80 | 115 | 0.343590 | 0.333333 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 257 | 1.000 | 0.353393 | 0.342716 | 0.010677 | 123 | 134 | 0.428016 | 0.420233 |

## Active Smoke/Inferno Intervals

- `4.0s` - `58.5s`, rows `110`
- `64.5s` - `106.5s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.5`, LSTM `0.1697`, XGBoost `0.3835`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.1836`, XGBoost `0.3835`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.1981`, XGBoost `0.3835`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.2084`, XGBoost `0.3878`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.2339`, XGBoost `0.3978`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.0578`, XGBoost `0.2193`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.2479`, XGBoost `0.3976`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.2509`, XGBoost `0.3995`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.0988`, XGBoost `0.2441`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.4038`, XGBoost `0.5020`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
