# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `17`
- rows: `203`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 203 | 1.000 | 0.169861 | 0.190013 | -0.020152 | 176 | 27 | 0.965517 | 1.000000 |
| active/recent utility | 203 | 1.000 | 0.169861 | 0.190013 | -0.020152 | 176 | 27 | 0.965517 | 1.000000 |
| strong utility action | 177 | 0.872 | 0.140835 | 0.161540 | -0.020704 | 154 | 23 | 0.960452 | 1.000000 |
| utility damage | 20 | 0.099 | 0.244401 | 0.317630 | -0.073229 | 17 | 3 | 1.000000 | 1.000000 |
| active smoke/inferno | 177 | 0.872 | 0.140835 | 0.161540 | -0.020704 | 154 | 23 | 0.960452 | 1.000000 |
| recent utility last 5s | 10 | 0.049 | 0.193706 | 0.299471 | -0.105765 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 203 | 1.000 | 0.169861 | 0.190013 | -0.020152 | 176 | 27 | 0.965517 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `55.0s`, rows `90`
- `56.5s` - `99.5s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.5`, LSTM `0.1746`, XGBoost `0.3008`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `44.0`, LSTM `0.1693`, XGBoost `0.2935`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `45.0`, LSTM `0.1801`, XGBoost `0.3024`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `43.5`, LSTM `0.1830`, XGBoost `0.2972`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `43.0`, LSTM `0.1957`, XGBoost `0.3045`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `24.5`, LSTM `0.1893`, XGBoost `0.2970`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.1958`, XGBoost `0.3025`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.1966`, XGBoost `0.3025`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.1911`, XGBoost `0.2967`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.1980`, XGBoost `0.3023`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `49.0`, recent_utility `0`
