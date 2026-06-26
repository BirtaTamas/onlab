# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `25`
- rows: `232`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 232 | 1.000 | 0.500397 | 0.525232 | -0.024836 | 100 | 132 | 0.750000 | 0.775862 |
| active/recent utility | 232 | 1.000 | 0.500397 | 0.525232 | -0.024836 | 100 | 132 | 0.750000 | 0.775862 |
| strong utility action | 161 | 0.694 | 0.552910 | 0.550077 | 0.002833 | 89 | 72 | 0.968944 | 0.968944 |
| utility damage | 10 | 0.043 | 0.593253 | 0.551542 | 0.041711 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 150 | 0.647 | 0.553783 | 0.549416 | 0.004367 | 88 | 62 | 0.966667 | 0.966667 |
| recent utility last 5s | 10 | 0.043 | 0.535004 | 0.559849 | -0.024844 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 232 | 1.000 | 0.500397 | 0.525232 | -0.024836 | 100 | 132 | 0.750000 | 0.775862 |

## Active Smoke/Inferno Intervals

- `7.5s` - `75.0s`, rows `136`
- `76.0s` - `82.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.5`, LSTM `0.1024`, XGBoost `0.1884`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.3362`, XGBoost `0.2537`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.6946`, XGBoost `0.7611`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.5003`, XGBoost `0.5662`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.1261`, XGBoost `0.1878`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.6915`, XGBoost `0.7526`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.5127`, XGBoost `0.5658`, closer `xgboost`, smoke `1`, inferno `4`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.6011`, XGBoost `0.5515`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `17.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.5993`, XGBoost `0.5515`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `17.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.3018`, XGBoost `0.2544`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
