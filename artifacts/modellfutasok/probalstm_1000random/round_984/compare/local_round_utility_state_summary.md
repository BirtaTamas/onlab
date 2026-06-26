# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `10`
- rows: `216`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 216 | 1.000 | 0.452362 | 0.493839 | -0.041477 | 73 | 143 | 0.606481 | 0.689815 |
| active/recent utility | 216 | 1.000 | 0.452362 | 0.493839 | -0.041477 | 73 | 143 | 0.606481 | 0.689815 |
| strong utility action | 144 | 0.667 | 0.506730 | 0.503008 | 0.003722 | 68 | 76 | 0.770833 | 0.812500 |
| utility damage | 10 | 0.046 | 0.514381 | 0.518042 | -0.003662 | 2 | 8 | 1.000000 | 1.000000 |
| active smoke/inferno | 144 | 0.667 | 0.506730 | 0.503008 | 0.003722 | 68 | 76 | 0.770833 | 0.812500 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 216 | 1.000 | 0.452362 | 0.493839 | -0.041477 | 73 | 143 | 0.606481 | 0.689815 |

## Active Smoke/Inferno Intervals

- `8.5s` - `56.5s`, rows `97`
- `58.0s` - `81.0s`, rows `47`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `73.0`, LSTM `0.5203`, XGBoost `0.3450`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.5152`, XGBoost `0.3419`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.5131`, XGBoost `0.3444`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.5096`, XGBoost `0.3416`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.5028`, XGBoost `0.3429`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.4973`, XGBoost `0.3460`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.5476`, XGBoost `0.4369`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.5221`, XGBoost `0.4361`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.5146`, XGBoost `0.4369`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5195`, XGBoost `0.4434`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
