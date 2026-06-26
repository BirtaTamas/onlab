# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `11`
- rows: `194`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 194 | 1.000 | 0.200437 | 0.332901 | -0.132464 | 194 | 0 | 0.979381 | 0.948454 |
| active/recent utility | 194 | 1.000 | 0.200437 | 0.332901 | -0.132464 | 194 | 0 | 0.979381 | 0.948454 |
| strong utility action | 160 | 0.825 | 0.200906 | 0.346421 | -0.145515 | 160 | 0 | 0.975000 | 0.937500 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 160 | 0.825 | 0.200906 | 0.346421 | -0.145515 | 160 | 0 | 0.975000 | 0.937500 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 194 | 1.000 | 0.200437 | 0.332901 | -0.132464 | 194 | 0 | 0.979381 | 0.948454 |

## Active Smoke/Inferno Intervals

- `9.5s` - `89.0s`, rows `160`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `16.5`, LSTM `0.2462`, XGBoost `0.4933`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.2365`, XGBoost `0.4809`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.2055`, XGBoost `0.4467`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.2573`, XGBoost `0.4981`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.2567`, XGBoost `0.4959`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.2433`, XGBoost `0.4809`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.2504`, XGBoost `0.4868`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.2653`, XGBoost `0.4964`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.2677`, XGBoost `0.4964`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.2741`, XGBoost `0.4964`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
