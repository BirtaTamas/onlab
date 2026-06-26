# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `27`
- rows: `125`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 125 | 1.000 | 0.319743 | 0.334587 | -0.014844 | 89 | 36 | 0.536000 | 0.536000 |
| active/recent utility | 125 | 1.000 | 0.319743 | 0.334587 | -0.014844 | 89 | 36 | 0.536000 | 0.536000 |
| strong utility action | 108 | 0.864 | 0.281425 | 0.303012 | -0.021586 | 89 | 19 | 0.611111 | 0.611111 |
| utility damage | 11 | 0.088 | 0.617611 | 0.657190 | -0.039578 | 11 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 108 | 0.864 | 0.281425 | 0.303012 | -0.021586 | 89 | 19 | 0.611111 | 0.611111 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 125 | 1.000 | 0.319743 | 0.334587 | -0.014844 | 89 | 36 | 0.536000 | 0.536000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `37.5s`, rows `60`
- `38.5s` - `62.0s`, rows `48`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.6266`, XGBoost `0.8139`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.6382`, XGBoost `0.8141`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.6420`, XGBoost `0.8141`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.6424`, XGBoost `0.8141`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.6450`, XGBoost `0.8137`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.6498`, XGBoost `0.8098`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.6717`, XGBoost `0.8098`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.6746`, XGBoost `0.8098`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.4211`, XGBoost `0.3023`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.4151`, XGBoost `0.3023`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
