# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv`
- round_num: `5`
- rows: `231`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 231 | 1.000 | 0.179379 | 0.300962 | -0.121583 | 197 | 34 | 0.939394 | 0.926407 |
| active/recent utility | 231 | 1.000 | 0.179379 | 0.300962 | -0.121583 | 197 | 34 | 0.939394 | 0.926407 |
| strong utility action | 163 | 0.706 | 0.215431 | 0.342983 | -0.127552 | 129 | 34 | 0.914110 | 0.895706 |
| utility damage | 22 | 0.095 | 0.324880 | 0.447443 | -0.122563 | 12 | 10 | 0.818182 | 0.818182 |
| active smoke/inferno | 161 | 0.697 | 0.215455 | 0.341469 | -0.126015 | 127 | 34 | 0.913043 | 0.900621 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 231 | 1.000 | 0.179379 | 0.300962 | -0.121583 | 197 | 34 | 0.939394 | 0.926407 |

## Active Smoke/Inferno Intervals

- `6.0s` - `54.5s`, rows `98`
- `65.5s` - `96.5s`, rows `63`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `93.5`, LSTM `0.2433`, XGBoost `0.5691`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `35.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.2452`, XGBoost `0.5667`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `35.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.2328`, XGBoost `0.5279`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `35.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.2488`, XGBoost `0.5325`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `27.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.2104`, XGBoost `0.4926`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `35.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.1867`, XGBoost `0.4555`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `35.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.1889`, XGBoost `0.4555`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `35.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.0491`, XGBoost `0.3095`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.0496`, XGBoost `0.3095`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.0519`, XGBoost `0.3095`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
