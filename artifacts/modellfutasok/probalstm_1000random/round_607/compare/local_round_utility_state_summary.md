# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-falcons-vs-mouz-bo3-plkh_Ps38mI3o_rFlgAljz/falcons-vs-mouz-m3-nuke-p3.csv`
- round_num: `6`
- rows: `222`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 222 | 1.000 | 0.671478 | 0.689373 | -0.017895 | 86 | 136 | 0.720721 | 0.720721 |
| active/recent utility | 222 | 1.000 | 0.671478 | 0.689373 | -0.017895 | 86 | 136 | 0.720721 | 0.720721 |
| strong utility action | 83 | 0.374 | 0.564009 | 0.552856 | 0.011152 | 55 | 28 | 0.506024 | 0.506024 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 83 | 0.374 | 0.564009 | 0.552856 | 0.011152 | 55 | 28 | 0.506024 | 0.506024 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 222 | 1.000 | 0.671478 | 0.689373 | -0.017895 | 86 | 136 | 0.720721 | 0.720721 |

## Active Smoke/Inferno Intervals

- `10.5s` - `37.5s`, rows `55`
- `47.5s` - `54.0s`, rows `14`
- `83.5s` - `90.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.0`, LSTM `0.6306`, XGBoost `0.5255`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.8930`, XGBoost `0.7985`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.8929`, XGBoost `0.7999`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5948`, XGBoost `0.5102`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.3963`, XGBoost `0.3152`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.3881`, XGBoost `0.3148`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.2405`, XGBoost `0.3135`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.7099`, XGBoost `0.7791`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.7116`, XGBoost `0.7791`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.7119`, XGBoost `0.7791`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
