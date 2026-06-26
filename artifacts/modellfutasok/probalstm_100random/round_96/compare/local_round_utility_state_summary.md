# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `13`
- rows: `139`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 139 | 1.000 | 0.515650 | 0.550133 | -0.034484 | 54 | 85 | 0.172662 | 0.143885 |
| active/recent utility | 139 | 1.000 | 0.515650 | 0.550133 | -0.034484 | 54 | 85 | 0.172662 | 0.143885 |
| strong utility action | 64 | 0.460 | 0.519399 | 0.601349 | -0.081949 | 47 | 17 | 0.375000 | 0.218750 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 64 | 0.460 | 0.519399 | 0.601349 | -0.081949 | 47 | 17 | 0.375000 | 0.218750 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 139 | 1.000 | 0.515650 | 0.550133 | -0.034484 | 54 | 85 | 0.172662 | 0.143885 |

## Active Smoke/Inferno Intervals

- `37.5s` - `69.0s`, rows `64`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.0`, LSTM `0.5760`, XGBoost `0.8621`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.6074`, XGBoost `0.8621`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.6000`, XGBoost `0.8407`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.6053`, XGBoost `0.8407`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.5396`, XGBoost `0.7577`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.5443`, XGBoost `0.7559`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5464`, XGBoost `0.7565`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.5847`, XGBoost `0.7859`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.6485`, XGBoost `0.8407`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.5608`, XGBoost `0.7525`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
