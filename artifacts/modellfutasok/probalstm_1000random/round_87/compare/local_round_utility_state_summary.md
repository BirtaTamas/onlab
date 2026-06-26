# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `20`
- rows: `265`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 265 | 1.000 | 0.372602 | 0.366881 | 0.005721 | 141 | 124 | 0.562264 | 0.535849 |
| active/recent utility | 265 | 1.000 | 0.372602 | 0.366881 | 0.005721 | 141 | 124 | 0.562264 | 0.535849 |
| strong utility action | 154 | 0.581 | 0.479540 | 0.449348 | 0.030192 | 58 | 96 | 0.435065 | 0.474026 |
| utility damage | 12 | 0.045 | 0.542612 | 0.600075 | -0.057462 | 12 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 154 | 0.581 | 0.479540 | 0.449348 | 0.030192 | 58 | 96 | 0.435065 | 0.474026 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 265 | 1.000 | 0.372602 | 0.366881 | 0.005721 | 141 | 124 | 0.562264 | 0.535849 |

## Active Smoke/Inferno Intervals

- `8.0s` - `36.5s`, rows `58`
- `50.0s` - `90.5s`, rows `82`
- `92.0s` - `98.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `61.0`, LSTM `0.5148`, XGBoost `0.3586`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `43.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.5066`, XGBoost `0.3581`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `43.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.5053`, XGBoost `0.3590`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `43.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5023`, XGBoost `0.3675`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.5019`, XGBoost `0.3675`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.5021`, XGBoost `0.3683`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.4992`, XGBoost `0.3656`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.5009`, XGBoost `0.3675`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.5016`, XGBoost `0.3683`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.4917`, XGBoost `0.3587`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `43.0`, recent_utility `0`
