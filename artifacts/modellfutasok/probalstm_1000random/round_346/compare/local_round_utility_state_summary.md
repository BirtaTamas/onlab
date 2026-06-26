# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `23`
- rows: `241`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 241 | 1.000 | 0.391796 | 0.455110 | -0.063314 | 226 | 15 | 0.755187 | 0.522822 |
| active/recent utility | 241 | 1.000 | 0.391796 | 0.455110 | -0.063314 | 226 | 15 | 0.755187 | 0.522822 |
| strong utility action | 220 | 0.913 | 0.402915 | 0.458274 | -0.055359 | 205 | 15 | 0.731818 | 0.500000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 208 | 0.863 | 0.397072 | 0.455378 | -0.058306 | 199 | 9 | 0.754808 | 0.514423 |
| recent utility last 5s | 12 | 0.050 | 0.504196 | 0.508482 | -0.004286 | 6 | 6 | 0.333333 | 0.250000 |
| flash effect present | 241 | 1.000 | 0.391796 | 0.455110 | -0.063314 | 226 | 15 | 0.755187 | 0.522822 |

## Active Smoke/Inferno Intervals

- `7.5s` - `34.0s`, rows `54`
- `40.0s` - `116.5s`, rows `154`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `94.5`, LSTM `0.5728`, XGBoost `0.7303`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.3521`, XGBoost `0.5096`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.5755`, XGBoost `0.7303`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.5795`, XGBoost `0.7303`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.5793`, XGBoost `0.7292`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.5821`, XGBoost `0.7303`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.3538`, XGBoost `0.4978`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.5664`, XGBoost `0.7096`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.5721`, XGBoost `0.7145`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.5690`, XGBoost `0.7103`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
