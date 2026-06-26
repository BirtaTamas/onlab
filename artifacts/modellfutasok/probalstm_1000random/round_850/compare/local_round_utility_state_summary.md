# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-vitality-bo3-ZpOL0o26IrRvvgFRbFxVou/lynn-vision-vs-vitality-m1-dust2.csv`
- round_num: `11`
- rows: `216`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 216 | 1.000 | 0.809042 | 0.764346 | 0.044695 | 179 | 37 | 1.000000 | 1.000000 |
| active/recent utility | 216 | 1.000 | 0.809042 | 0.764346 | 0.044695 | 179 | 37 | 1.000000 | 1.000000 |
| strong utility action | 207 | 0.958 | 0.811020 | 0.770620 | 0.040400 | 170 | 37 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.046 | 0.700864 | 0.605797 | 0.095067 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 198 | 0.917 | 0.815076 | 0.778304 | 0.036772 | 161 | 37 | 1.000000 | 1.000000 |
| recent utility last 5s | 11 | 0.051 | 0.717060 | 0.601564 | 0.115496 | 11 | 0 | 1.000000 | 1.000000 |
| flash effect present | 216 | 1.000 | 0.809042 | 0.764346 | 0.044695 | 179 | 37 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `107.5s`, rows `198`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `4.5`, LSTM `0.7406`, XGBoost `0.6064`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.7356`, XGBoost `0.6015`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.0`, LSTM `0.7236`, XGBoost `0.6015`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.7203`, XGBoost `0.6015`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.5`, LSTM `0.7191`, XGBoost `0.6015`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.5`, LSTM `0.7164`, XGBoost `0.6000`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.5`, LSTM `0.7151`, XGBoost `0.6000`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `16.0`, LSTM `0.7199`, XGBoost `0.6055`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.7142`, XGBoost `0.6015`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.0`, LSTM `0.7110`, XGBoost `0.6000`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
