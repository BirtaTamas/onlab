# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-gamerlegion-vs-tyloo-bo3-CHuj0-KFwAe9c3Zh96vlUq/gamerlegion-vs-tyloo-m2-ancient.csv`
- round_num: `1`
- rows: `163`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 163 | 1.000 | 0.681509 | 0.721286 | -0.039777 | 13 | 150 | 0.822086 | 0.957055 |
| active/recent utility | 163 | 1.000 | 0.681509 | 0.721286 | -0.039777 | 13 | 150 | 0.822086 | 0.957055 |
| strong utility action | 55 | 0.337 | 0.501593 | 0.526879 | -0.025286 | 13 | 42 | 0.618182 | 0.872727 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.270 | 0.496271 | 0.529076 | -0.032805 | 4 | 40 | 0.522727 | 0.840909 |
| recent utility last 5s | 11 | 0.067 | 0.522880 | 0.518091 | 0.004789 | 9 | 2 | 1.000000 | 1.000000 |
| flash effect present | 163 | 1.000 | 0.681509 | 0.721286 | -0.039777 | 13 | 150 | 0.822086 | 0.957055 |

## Active Smoke/Inferno Intervals

- `23.5s` - `45.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.5`, LSTM `0.6809`, XGBoost `0.8771`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.7117`, XGBoost `0.8765`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.7363`, XGBoost `0.8654`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.3378`, XGBoost `0.2131`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.7537`, XGBoost `0.8664`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.4143`, XGBoost `0.5113`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.4579`, XGBoost `0.5232`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.4658`, XGBoost `0.5232`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5274`, XGBoost `0.4707`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.4450`, XGBoost `0.5012`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
