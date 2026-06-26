# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `2`
- rows: `187`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 187 | 1.000 | 0.164210 | 0.229318 | -0.065108 | 176 | 11 | 1.000000 | 1.000000 |
| active/recent utility | 187 | 1.000 | 0.164210 | 0.229318 | -0.065108 | 176 | 11 | 1.000000 | 1.000000 |
| strong utility action | 175 | 0.936 | 0.162917 | 0.228723 | -0.065806 | 165 | 10 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.053 | 0.159722 | 0.237867 | -0.078145 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 160 | 0.856 | 0.158467 | 0.228467 | -0.070000 | 159 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.053 | 0.239095 | 0.228163 | 0.010932 | 1 | 9 | 1.000000 | 1.000000 |
| flash effect present | 187 | 1.000 | 0.164210 | 0.229318 | -0.065108 | 176 | 11 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `37.5s`, rows `60`
- `43.5s` - `93.0s`, rows `100`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `66.0`, LSTM `0.1388`, XGBoost `0.3190`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `24.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.1090`, XGBoost `0.2880`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `27.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.1106`, XGBoost `0.2890`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `17.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.1464`, XGBoost `0.3172`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `24.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.1231`, XGBoost `0.2877`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `27.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.1574`, XGBoost `0.3190`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.1321`, XGBoost `0.2933`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.1649`, XGBoost `0.3185`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `24.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.1633`, XGBoost `0.3168`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `24.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.1663`, XGBoost `0.3185`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `24.0`, recent_utility `0`
