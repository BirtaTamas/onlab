# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m2-mirage.csv`
- round_num: `14`
- rows: `123`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 123 | 1.000 | 0.340207 | 0.372019 | -0.031812 | 92 | 31 | 0.829268 | 0.853659 |
| active/recent utility | 123 | 1.000 | 0.340207 | 0.372019 | -0.031812 | 92 | 31 | 0.829268 | 0.853659 |
| strong utility action | 122 | 0.992 | 0.341150 | 0.372270 | -0.031120 | 91 | 31 | 0.827869 | 0.852459 |
| utility damage | 10 | 0.081 | 0.458575 | 0.430719 | 0.027856 | 6 | 4 | 0.700000 | 1.000000 |
| active smoke/inferno | 108 | 0.878 | 0.338367 | 0.373449 | -0.035082 | 85 | 23 | 0.805556 | 0.833333 |
| recent utility last 5s | 14 | 0.114 | 0.362616 | 0.363175 | -0.000558 | 6 | 8 | 1.000000 | 1.000000 |
| flash effect present | 123 | 1.000 | 0.340207 | 0.372019 | -0.031812 | 92 | 31 | 0.829268 | 0.853659 |

## Active Smoke/Inferno Intervals

- `7.5s` - `61.0s`, rows `108`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.0`, LSTM `0.8104`, XGBoost `0.6106`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `24.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.2260`, XGBoost `0.3980`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.7668`, XGBoost `0.6077`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `24.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.7741`, XGBoost `0.6191`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `24.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.2142`, XGBoost `0.3670`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.2541`, XGBoost `0.4017`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.7647`, XGBoost `0.6183`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `24.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.2228`, XGBoost `0.3670`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.2239`, XGBoost `0.3670`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.2330`, XGBoost `0.3677`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
