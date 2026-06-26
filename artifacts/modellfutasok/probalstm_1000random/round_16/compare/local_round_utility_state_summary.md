# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-gamerlegion-vs-tyloo-bo3-CHuj0-KFwAe9c3Zh96vlUq/gamerlegion-vs-tyloo-m2-ancient.csv`
- round_num: `12`
- rows: `177`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 177 | 1.000 | 0.834826 | 0.839641 | -0.004816 | 57 | 120 | 1.000000 | 1.000000 |
| active/recent utility | 177 | 1.000 | 0.834826 | 0.839641 | -0.004816 | 57 | 120 | 1.000000 | 1.000000 |
| strong utility action | 158 | 0.893 | 0.846787 | 0.856701 | -0.009914 | 45 | 113 | 1.000000 | 1.000000 |
| utility damage | 17 | 0.096 | 0.590841 | 0.564813 | 0.026029 | 16 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 158 | 0.893 | 0.846787 | 0.856701 | -0.009914 | 45 | 113 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 177 | 1.000 | 0.834826 | 0.839641 | -0.004816 | 57 | 120 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `84.5s`, rows `158`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `6.0`, LSTM `0.6041`, XGBoost `0.5196`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.6029`, XGBoost `0.5203`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6178`, XGBoost `0.5485`, closer `lstm`, smoke `4`, inferno `4`, utility_damage `31.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.8433`, XGBoost `0.9115`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `52.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.5941`, XGBoost `0.5299`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.8487`, XGBoost `0.9118`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.8538`, XGBoost `0.9141`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.5880`, XGBoost `0.5299`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.8539`, XGBoost `0.9118`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `52.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.6238`, XGBoost `0.5660`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
