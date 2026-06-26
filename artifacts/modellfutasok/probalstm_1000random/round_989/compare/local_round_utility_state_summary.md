# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `19`
- rows: `298`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 298 | 1.000 | 0.193401 | 0.214959 | -0.021558 | 229 | 69 | 0.996644 | 1.000000 |
| active/recent utility | 298 | 1.000 | 0.193401 | 0.214959 | -0.021558 | 229 | 69 | 0.996644 | 1.000000 |
| strong utility action | 221 | 0.742 | 0.248807 | 0.274804 | -0.025997 | 172 | 49 | 0.995475 | 1.000000 |
| utility damage | 20 | 0.067 | 0.380735 | 0.345963 | 0.034772 | 4 | 16 | 1.000000 | 1.000000 |
| active smoke/inferno | 211 | 0.708 | 0.246262 | 0.271472 | -0.025210 | 163 | 48 | 0.995261 | 1.000000 |
| recent utility last 5s | 20 | 0.067 | 0.294113 | 0.328929 | -0.034816 | 19 | 1 | 1.000000 | 1.000000 |
| flash effect present | 298 | 1.000 | 0.193401 | 0.214959 | -0.021558 | 229 | 69 | 0.996644 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `56.0s`, rows `98`
- `58.5s` - `114.5s`, rows `113`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `77.5`, LSTM `0.5008`, XGBoost `0.3509`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.4961`, XGBoost `0.3517`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.4778`, XGBoost `0.3438`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.4845`, XGBoost `0.3509`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.4831`, XGBoost `0.3517`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.4741`, XGBoost `0.3442`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.4807`, XGBoost `0.3517`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.4741`, XGBoost `0.3476`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.2385`, XGBoost `0.3594`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.4710`, XGBoost `0.3509`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
