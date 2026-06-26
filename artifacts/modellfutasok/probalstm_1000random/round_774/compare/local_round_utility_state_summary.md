# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `1`
- rows: `159`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 159 | 1.000 | 0.699949 | 0.752263 | -0.052314 | 20 | 139 | 0.798742 | 0.836478 |
| active/recent utility | 159 | 1.000 | 0.699949 | 0.752263 | -0.052314 | 20 | 139 | 0.798742 | 0.836478 |
| strong utility action | 55 | 0.346 | 0.728712 | 0.782175 | -0.053464 | 12 | 43 | 1.000000 | 0.800000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.277 | 0.781721 | 0.853365 | -0.071644 | 1 | 43 | 1.000000 | 1.000000 |
| recent utility last 5s | 11 | 0.069 | 0.516674 | 0.497416 | 0.019258 | 11 | 0 | 1.000000 | 0.000000 |
| flash effect present | 159 | 1.000 | 0.699949 | 0.752263 | -0.052314 | 20 | 139 | 0.798742 | 0.836478 |

## Active Smoke/Inferno Intervals

- `37.5s` - `59.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.5`, LSTM `0.5626`, XGBoost `0.7451`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.5761`, XGBoost `0.7451`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.5854`, XGBoost `0.7459`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.5911`, XGBoost `0.7459`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.5916`, XGBoost `0.7455`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.5938`, XGBoost `0.7450`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.5966`, XGBoost `0.7453`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5967`, XGBoost `0.7450`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5994`, XGBoost `0.7444`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.6056`, XGBoost `0.7450`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
