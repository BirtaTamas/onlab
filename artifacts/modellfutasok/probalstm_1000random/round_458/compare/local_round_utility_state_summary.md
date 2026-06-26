# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `9`
- rows: `208`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 208 | 1.000 | 0.730479 | 0.743153 | -0.012674 | 123 | 85 | 0.038462 | 0.000000 |
| active/recent utility | 208 | 1.000 | 0.730479 | 0.743153 | -0.012674 | 123 | 85 | 0.038462 | 0.000000 |
| strong utility action | 155 | 0.745 | 0.754890 | 0.764776 | -0.009886 | 94 | 61 | 0.000000 | 0.000000 |
| utility damage | 24 | 0.115 | 0.762195 | 0.731805 | 0.030390 | 11 | 13 | 0.000000 | 0.000000 |
| active smoke/inferno | 155 | 0.745 | 0.754890 | 0.764776 | -0.009886 | 94 | 61 | 0.000000 | 0.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 208 | 1.000 | 0.730479 | 0.743153 | -0.012674 | 123 | 85 | 0.038462 | 0.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `17.0s`, rows `16`
- `19.0s` - `24.0s`, rows `11`
- `26.5s` - `90.0s`, rows `128`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `85.5`, LSTM `0.6735`, XGBoost `0.8574`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.6772`, XGBoost `0.8554`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.6857`, XGBoost `0.8555`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.7026`, XGBoost `0.8555`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.7095`, XGBoost `0.8511`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.6966`, XGBoost `0.5609`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.7207`, XGBoost `0.8486`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.7259`, XGBoost `0.8511`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6828`, XGBoost `0.5609`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.7308`, XGBoost `0.8491`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
