# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `11`
- rows: `214`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 214 | 1.000 | 0.432109 | 0.521748 | -0.089638 | 2 | 212 | 0.336449 | 0.635514 |
| active/recent utility | 214 | 1.000 | 0.432109 | 0.521748 | -0.089638 | 2 | 212 | 0.336449 | 0.635514 |
| strong utility action | 166 | 0.776 | 0.484542 | 0.567658 | -0.083116 | 2 | 164 | 0.433735 | 0.716867 |
| utility damage | 10 | 0.047 | 0.496372 | 0.556268 | -0.059896 | 0 | 10 | 0.300000 | 1.000000 |
| active smoke/inferno | 166 | 0.776 | 0.484542 | 0.567658 | -0.083116 | 2 | 164 | 0.433735 | 0.716867 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 214 | 1.000 | 0.432109 | 0.521748 | -0.089638 | 2 | 212 | 0.336449 | 0.635514 |

## Active Smoke/Inferno Intervals

- `8.5s` - `72.5s`, rows `129`
- `88.5s` - `106.5s`, rows `37`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.0`, LSTM `0.5674`, XGBoost `0.7371`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5689`, XGBoost `0.7356`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.0316`, XGBoost `0.1982`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.5710`, XGBoost `0.7371`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5696`, XGBoost `0.7342`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.0355`, XGBoost `0.1966`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.0307`, XGBoost `0.1912`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.0391`, XGBoost `0.1982`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.0393`, XGBoost `0.1966`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5802`, XGBoost `0.7374`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
