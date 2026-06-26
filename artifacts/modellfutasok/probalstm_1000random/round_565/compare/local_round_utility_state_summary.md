# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-tyloo-vs-vitality-bo3-aF98ikh3PjdqKlkdIJn9tC/tyloo-vs-vitality-m1-inferno.csv`
- round_num: `16`
- rows: `183`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 183 | 1.000 | 0.200417 | 0.190299 | 0.010118 | 114 | 69 | 0.726776 | 0.819672 |
| active/recent utility | 183 | 1.000 | 0.200417 | 0.190299 | 0.010118 | 114 | 69 | 0.726776 | 0.819672 |
| strong utility action | 107 | 0.585 | 0.239015 | 0.222000 | 0.017015 | 56 | 51 | 0.710280 | 0.869159 |
| utility damage | 10 | 0.055 | 0.333302 | 0.275843 | 0.057459 | 3 | 7 | 0.800000 | 0.900000 |
| active smoke/inferno | 107 | 0.585 | 0.239015 | 0.222000 | 0.017015 | 56 | 51 | 0.710280 | 0.869159 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 183 | 1.000 | 0.200417 | 0.190299 | 0.010118 | 114 | 69 | 0.726776 | 0.819672 |

## Active Smoke/Inferno Intervals

- `9.5s` - `62.5s`, rows `107`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `23.5`, LSTM `0.5802`, XGBoost `0.4112`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.3911`, XGBoost `0.2274`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `29.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.3498`, XGBoost `0.2113`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.3499`, XGBoost `0.2127`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.3534`, XGBoost `0.2180`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.3359`, XGBoost `0.2020`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.3490`, XGBoost `0.2175`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.3474`, XGBoost `0.2159`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.3486`, XGBoost `0.2174`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.3433`, XGBoost `0.2127`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
