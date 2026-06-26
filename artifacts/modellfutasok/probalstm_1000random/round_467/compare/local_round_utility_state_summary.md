# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m3-dust2.csv`
- round_num: `3`
- rows: `204`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 204 | 1.000 | 0.052555 | 0.145945 | -0.093390 | 203 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 204 | 1.000 | 0.052555 | 0.145945 | -0.093390 | 203 | 1 | 1.000000 | 1.000000 |
| strong utility action | 181 | 0.887 | 0.055517 | 0.155247 | -0.099729 | 181 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 177 | 0.868 | 0.052923 | 0.152304 | -0.099381 | 177 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 18 | 0.088 | 0.165664 | 0.283086 | -0.117422 | 18 | 0 | 1.000000 | 1.000000 |
| flash effect present | 204 | 1.000 | 0.052555 | 0.145945 | -0.093390 | 203 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `2.5s` - `90.5s`, rows `177`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `32.5`, LSTM `0.0435`, XGBoost `0.3312`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.0449`, XGBoost `0.3312`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0495`, XGBoost `0.3312`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0485`, XGBoost `0.3237`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0560`, XGBoost `0.3237`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.0591`, XGBoost `0.3237`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0634`, XGBoost `0.3237`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0622`, XGBoost `0.3169`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0693`, XGBoost `0.3237`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.0768`, XGBoost `0.3237`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
