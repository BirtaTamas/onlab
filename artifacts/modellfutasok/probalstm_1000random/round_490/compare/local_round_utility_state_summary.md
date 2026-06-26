# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `5`
- rows: `261`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 261 | 1.000 | 0.044869 | 0.054507 | -0.009637 | 144 | 117 | 1.000000 | 1.000000 |
| active/recent utility | 261 | 1.000 | 0.044869 | 0.054507 | -0.009637 | 144 | 117 | 1.000000 | 1.000000 |
| strong utility action | 200 | 0.766 | 0.053815 | 0.063526 | -0.009711 | 120 | 80 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.038 | 0.237639 | 0.256419 | -0.018780 | 8 | 2 | 1.000000 | 1.000000 |
| active smoke/inferno | 185 | 0.709 | 0.041456 | 0.046852 | -0.005396 | 105 | 80 | 1.000000 | 1.000000 |
| recent utility last 5s | 15 | 0.057 | 0.206238 | 0.269169 | -0.062930 | 15 | 0 | 1.000000 | 1.000000 |
| flash effect present | 261 | 1.000 | 0.044869 | 0.054507 | -0.009637 | 144 | 117 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `49.0s`, rows `79`
- `52.0s` - `58.5s`, rows `14`
- `59.5s` - `105.0s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.5`, LSTM `0.1037`, XGBoost `0.2627`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.1212`, XGBoost `0.2732`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `1.0`, LSTM `0.0917`, XGBoost `0.2080`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.1000`, XGBoost `0.2080`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.5`, LSTM `0.1897`, XGBoost `0.2795`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.2025`, XGBoost `0.2829`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `10.0`, LSTM `0.2072`, XGBoost `0.2804`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.2278`, XGBoost `0.2946`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.2334`, XGBoost `0.2946`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.2357`, XGBoost `0.2942`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
