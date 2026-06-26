# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m3-ancient.csv`
- round_num: `11`
- rows: `173`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.221027 | 0.251829 | -0.030801 | 121 | 52 | 0.803468 | 0.838150 |
| active/recent utility | 173 | 1.000 | 0.221027 | 0.251829 | -0.030801 | 121 | 52 | 0.803468 | 0.838150 |
| strong utility action | 149 | 0.861 | 0.240735 | 0.277738 | -0.037003 | 101 | 48 | 0.798658 | 0.838926 |
| utility damage | 20 | 0.116 | 0.501191 | 0.446887 | 0.054304 | 1 | 19 | 0.500000 | 0.500000 |
| active smoke/inferno | 140 | 0.809 | 0.218552 | 0.262333 | -0.043782 | 101 | 39 | 0.850000 | 0.892857 |
| recent utility last 5s | 10 | 0.058 | 0.584104 | 0.516497 | 0.067607 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 173 | 1.000 | 0.221027 | 0.251829 | -0.030801 | 121 | 52 | 0.803468 | 0.838150 |

## Active Smoke/Inferno Intervals

- `6.5s` - `76.0s`, rows `140`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.5`, LSTM `0.1467`, XGBoost `0.3013`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.1158`, XGBoost `0.2674`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0762`, XGBoost `0.2256`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.0870`, XGBoost `0.2341`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.0793`, XGBoost `0.2256`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.0789`, XGBoost `0.2250`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.0776`, XGBoost `0.2236`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.4963`, XGBoost `0.3522`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.0805`, XGBoost `0.2239`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.0863`, XGBoost `0.2292`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
