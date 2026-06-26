# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m2-ancient.csv`
- round_num: `13`
- rows: `102`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 102 | 1.000 | 0.510485 | 0.541306 | -0.030820 | 26 | 76 | 0.490196 | 0.372549 |
| active/recent utility | 102 | 1.000 | 0.510485 | 0.541306 | -0.030820 | 26 | 76 | 0.490196 | 0.372549 |
| strong utility action | 55 | 0.539 | 0.492465 | 0.524489 | -0.032024 | 12 | 43 | 0.363636 | 0.327273 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 45 | 0.441 | 0.491623 | 0.527727 | -0.036104 | 9 | 36 | 0.311111 | 0.177778 |
| recent utility last 5s | 10 | 0.098 | 0.496254 | 0.509916 | -0.013662 | 3 | 7 | 0.600000 | 1.000000 |
| flash effect present | 102 | 1.000 | 0.510485 | 0.541306 | -0.030820 | 26 | 76 | 0.490196 | 0.372549 |

## Active Smoke/Inferno Intervals

- `25.0s` - `47.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `42.0`, LSTM `0.2979`, XGBoost `0.4411`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.6574`, XGBoost `0.7667`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.6415`, XGBoost `0.7458`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.6510`, XGBoost `0.7538`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.6614`, XGBoost `0.7640`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.3323`, XGBoost `0.4293`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.7823`, XGBoost `0.8705`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.6592`, XGBoost `0.7458`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.3571`, XGBoost `0.4338`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.6884`, XGBoost `0.7618`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
