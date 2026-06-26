# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-passion-ua-vs-spirit-bo3-WimU0hRkNcqhh3KAjCozBx/passion-ua-vs-spirit-m2-mirage.csv`
- round_num: `1`
- rows: `147`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 147 | 1.000 | 0.256545 | 0.318702 | -0.062158 | 147 | 0 | 0.619048 | 0.523810 |
| active/recent utility | 147 | 1.000 | 0.256545 | 0.318702 | -0.062158 | 147 | 0 | 0.619048 | 0.523810 |
| strong utility action | 72 | 0.490 | 0.405158 | 0.502482 | -0.097325 | 72 | 0 | 0.388889 | 0.236111 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 72 | 0.490 | 0.405158 | 0.502482 | -0.097325 | 72 | 0 | 0.388889 | 0.236111 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 147 | 1.000 | 0.256545 | 0.318702 | -0.062158 | 147 | 0 | 0.619048 | 0.523810 |

## Active Smoke/Inferno Intervals

- `7.5s` - `43.0s`, rows `72`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.5`, LSTM `0.2633`, XGBoost `0.5328`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.2731`, XGBoost `0.5328`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.2768`, XGBoost `0.5336`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.2810`, XGBoost `0.5331`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.2837`, XGBoost `0.5336`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.5603`, XGBoost `0.8034`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5632`, XGBoost `0.8034`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.2936`, XGBoost `0.5331`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.3147`, XGBoost `0.5336`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5902`, XGBoost `0.8034`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
