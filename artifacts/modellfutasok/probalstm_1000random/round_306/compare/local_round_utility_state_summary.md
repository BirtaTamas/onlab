# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `22`
- rows: `165`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 165 | 1.000 | 0.400722 | 0.554200 | -0.153478 | 157 | 8 | 0.569697 | 0.290909 |
| active/recent utility | 165 | 1.000 | 0.400722 | 0.554200 | -0.153478 | 157 | 8 | 0.569697 | 0.290909 |
| strong utility action | 75 | 0.455 | 0.571574 | 0.641238 | -0.069665 | 67 | 8 | 0.226667 | 0.146667 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 75 | 0.455 | 0.571574 | 0.641238 | -0.069665 | 67 | 8 | 0.226667 | 0.146667 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 165 | 1.000 | 0.400722 | 0.554200 | -0.153478 | 157 | 8 | 0.569697 | 0.290909 |

## Active Smoke/Inferno Intervals

- `6.5s` - `43.5s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.5`, LSTM `0.2461`, XGBoost `0.6586`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.2851`, XGBoost `0.6586`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.3231`, XGBoost `0.6498`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.3708`, XGBoost `0.6465`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.6328`, XGBoost `0.8562`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.6532`, XGBoost `0.8598`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.6617`, XGBoost `0.8515`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.6658`, XGBoost `0.8506`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.6700`, XGBoost `0.8506`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.6847`, XGBoost `0.8598`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
