# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `12`
- rows: `153`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 153 | 1.000 | 0.068937 | 0.144724 | -0.075787 | 153 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 153 | 1.000 | 0.068937 | 0.144724 | -0.075787 | 153 | 0 | 1.000000 | 1.000000 |
| strong utility action | 107 | 0.699 | 0.067037 | 0.142628 | -0.075592 | 107 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 107 | 0.699 | 0.067037 | 0.142628 | -0.075592 | 107 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 153 | 1.000 | 0.068937 | 0.144724 | -0.075787 | 153 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `62.0s`, rows `107`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `20.5`, LSTM `0.1465`, XGBoost `0.3442`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.1453`, XGBoost `0.3415`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.1426`, XGBoost `0.3377`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.1431`, XGBoost `0.3377`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.1470`, XGBoost `0.3415`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.1415`, XGBoost `0.3346`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1576`, XGBoost `0.3487`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.1462`, XGBoost `0.3355`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.1478`, XGBoost `0.3356`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.1484`, XGBoost `0.3356`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
