# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-tyloo-vs-vitality-bo3-aF98ikh3PjdqKlkdIJn9tC/tyloo-vs-vitality-m1-inferno.csv`
- round_num: `10`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.784593 | 0.781991 | 0.002601 | 81 | 149 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.784593 | 0.781991 | 0.002601 | 81 | 149 | 1.000000 | 1.000000 |
| strong utility action | 194 | 0.843 | 0.777836 | 0.777401 | 0.000435 | 65 | 129 | 1.000000 | 1.000000 |
| utility damage | 33 | 0.143 | 0.621078 | 0.541987 | 0.079091 | 29 | 4 | 1.000000 | 1.000000 |
| active smoke/inferno | 194 | 0.843 | 0.777836 | 0.777401 | 0.000435 | 65 | 129 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.784593 | 0.781991 | 0.002601 | 81 | 149 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `97.5s`, rows `183`
- `99.0s` - `104.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.0`, LSTM `0.6395`, XGBoost `0.5085`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `12.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.6477`, XGBoost `0.5170`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `23.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.6434`, XGBoost `0.5148`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `23.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6345`, XGBoost `0.5083`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6341`, XGBoost `0.5081`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `12.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6418`, XGBoost `0.5164`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `23.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.6373`, XGBoost `0.5143`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `23.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.6392`, XGBoost `0.5164`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `23.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6370`, XGBoost `0.5164`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `23.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6243`, XGBoost `0.5046`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
