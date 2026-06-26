# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `16`
- rows: `139`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 139 | 1.000 | 0.759378 | 0.801665 | -0.042287 | 38 | 101 | 1.000000 | 1.000000 |
| active/recent utility | 139 | 1.000 | 0.759378 | 0.801665 | -0.042287 | 38 | 101 | 1.000000 | 1.000000 |
| strong utility action | 130 | 0.935 | 0.766266 | 0.814010 | -0.047744 | 30 | 100 | 1.000000 | 1.000000 |
| utility damage | 39 | 0.281 | 0.764881 | 0.801185 | -0.036304 | 8 | 31 | 1.000000 | 1.000000 |
| active smoke/inferno | 120 | 0.863 | 0.776782 | 0.830287 | -0.053505 | 20 | 100 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.072 | 0.640075 | 0.618684 | 0.021392 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 139 | 1.000 | 0.759378 | 0.801665 | -0.042287 | 38 | 101 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `69.0s`, rows `120`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.0`, LSTM `0.6538`, XGBoost `0.8246`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.6693`, XGBoost `0.8225`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.6752`, XGBoost `0.8273`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.6810`, XGBoost `0.8246`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.6848`, XGBoost `0.8225`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.6924`, XGBoost `0.8296`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.6858`, XGBoost `0.8225`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.8299`, XGBoost `0.9650`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `8.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.7584`, XGBoost `0.8885`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.6893`, XGBoost `0.8191`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
