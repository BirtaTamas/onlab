# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `11`
- rows: `199`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 199 | 1.000 | 0.017078 | 0.036320 | -0.019241 | 174 | 25 | 1.000000 | 1.000000 |
| active/recent utility | 199 | 1.000 | 0.017078 | 0.036320 | -0.019241 | 174 | 25 | 1.000000 | 1.000000 |
| strong utility action | 134 | 0.673 | 0.021679 | 0.039114 | -0.017435 | 116 | 18 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 134 | 0.673 | 0.021679 | 0.039114 | -0.017435 | 116 | 18 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 199 | 1.000 | 0.017078 | 0.036320 | -0.019241 | 174 | 25 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `34.5s`, rows `51`
- `36.5s` - `70.5s`, rows `69`
- `81.5s` - `88.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.5`, LSTM `0.0238`, XGBoost `0.0875`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0252`, XGBoost `0.0884`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0261`, XGBoost `0.0885`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0266`, XGBoost `0.0880`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0302`, XGBoost `0.0885`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0419`, XGBoost `0.0987`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0446`, XGBoost `0.0989`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0448`, XGBoost `0.0984`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.0415`, XGBoost `0.0948`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.0587`, XGBoost `0.1108`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
