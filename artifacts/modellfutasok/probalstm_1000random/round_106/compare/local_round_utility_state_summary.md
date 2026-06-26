# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `3`
- rows: `181`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 181 | 1.000 | 0.584639 | 0.519396 | 0.065243 | 141 | 40 | 0.972376 | 0.569061 |
| active/recent utility | 181 | 1.000 | 0.584639 | 0.519396 | 0.065243 | 141 | 40 | 0.972376 | 0.569061 |
| strong utility action | 139 | 0.768 | 0.585337 | 0.524836 | 0.060502 | 102 | 37 | 0.985612 | 0.561151 |
| utility damage | 40 | 0.221 | 0.635941 | 0.571497 | 0.064445 | 28 | 12 | 0.950000 | 0.375000 |
| active smoke/inferno | 139 | 0.768 | 0.585337 | 0.524836 | 0.060502 | 102 | 37 | 0.985612 | 0.561151 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 181 | 1.000 | 0.584639 | 0.519396 | 0.065243 | 141 | 40 | 0.972376 | 0.569061 |

## Active Smoke/Inferno Intervals

- `10.5s` - `33.0s`, rows `46`
- `36.5s` - `77.0s`, rows `82`
- `85.0s` - `90.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.5`, LSTM `0.6154`, XGBoost `0.4147`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.6069`, XGBoost `0.4070`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6527`, XGBoost `0.4688`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.6733`, XGBoost `0.4938`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `104.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.6506`, XGBoost `0.4714`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `80.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.6693`, XGBoost `0.4910`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `104.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.6520`, XGBoost `0.4780`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `64.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.6407`, XGBoost `0.4688`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.6407`, XGBoost `0.4688`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.6402`, XGBoost `0.4688`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
