# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `29`
- rows: `127`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 127 | 1.000 | 0.579033 | 0.519274 | 0.059759 | 101 | 26 | 0.685039 | 0.677165 |
| active/recent utility | 127 | 1.000 | 0.579033 | 0.519274 | 0.059759 | 101 | 26 | 0.685039 | 0.677165 |
| strong utility action | 100 | 0.787 | 0.540901 | 0.462392 | 0.078509 | 87 | 13 | 0.600000 | 0.590000 |
| utility damage | 10 | 0.079 | 0.536129 | 0.505949 | 0.030179 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 100 | 0.787 | 0.540901 | 0.462392 | 0.078509 | 87 | 13 | 0.600000 | 0.590000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 127 | 1.000 | 0.579033 | 0.519274 | 0.059759 | 101 | 26 | 0.685039 | 0.677165 |

## Active Smoke/Inferno Intervals

- `8.0s` - `57.5s`, rows `100`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `32.0`, LSTM `0.4963`, XGBoost `0.2500`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.5191`, XGBoost `0.2805`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.4980`, XGBoost `0.2901`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.4954`, XGBoost `0.2890`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.4911`, XGBoost `0.2901`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.4868`, XGBoost `0.2901`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.4834`, XGBoost `0.2886`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.4845`, XGBoost `0.2901`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.4837`, XGBoost `0.2899`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.4804`, XGBoost `0.2867`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
