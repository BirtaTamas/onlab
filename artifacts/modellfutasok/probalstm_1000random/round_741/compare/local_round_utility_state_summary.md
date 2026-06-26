# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `3`
- rows: `146`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 146 | 1.000 | 0.005903 | 0.024239 | -0.018336 | 146 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 146 | 1.000 | 0.005903 | 0.024239 | -0.018336 | 146 | 0 | 1.000000 | 1.000000 |
| strong utility action | 98 | 0.671 | 0.006001 | 0.022778 | -0.016777 | 98 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 98 | 0.671 | 0.006001 | 0.022778 | -0.016777 | 98 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 146 | 1.000 | 0.005903 | 0.024239 | -0.018336 | 146 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `32.5s`, rows `54`
- `51.0s` - `72.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.0`, LSTM `0.0207`, XGBoost `0.0898`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0239`, XGBoost `0.0898`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0168`, XGBoost `0.0777`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.0138`, XGBoost `0.0743`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.0152`, XGBoost `0.0745`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.0157`, XGBoost `0.0745`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0173`, XGBoost `0.0746`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.0174`, XGBoost `0.0745`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0205`, XGBoost `0.0768`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0213`, XGBoost `0.0758`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
