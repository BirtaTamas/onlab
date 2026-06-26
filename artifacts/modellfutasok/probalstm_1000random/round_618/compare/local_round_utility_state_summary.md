# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `8`
- rows: `187`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 187 | 1.000 | 0.171107 | 0.147257 | 0.023849 | 75 | 112 | 0.754011 | 0.844920 |
| active/recent utility | 187 | 1.000 | 0.171107 | 0.147257 | 0.023849 | 75 | 112 | 0.754011 | 0.844920 |
| strong utility action | 131 | 0.701 | 0.190704 | 0.159598 | 0.031106 | 57 | 74 | 0.740458 | 0.870229 |
| utility damage | 10 | 0.053 | 0.473662 | 0.410093 | 0.063568 | 2 | 8 | 0.500000 | 1.000000 |
| active smoke/inferno | 131 | 0.701 | 0.190704 | 0.159598 | 0.031106 | 57 | 74 | 0.740458 | 0.870229 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 187 | 1.000 | 0.171107 | 0.147257 | 0.023849 | 75 | 112 | 0.754011 | 0.844920 |

## Active Smoke/Inferno Intervals

- `6.0s` - `71.0s`, rows `131`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.5`, LSTM `0.4090`, XGBoost `0.0987`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.3206`, XGBoost `0.1009`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.2694`, XGBoost `0.1068`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.2646`, XGBoost `0.1022`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.2447`, XGBoost `0.1093`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.2470`, XGBoost `0.1123`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.2413`, XGBoost `0.1123`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.2390`, XGBoost `0.1125`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5243`, XGBoost `0.3998`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5237`, XGBoost `0.3994`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
