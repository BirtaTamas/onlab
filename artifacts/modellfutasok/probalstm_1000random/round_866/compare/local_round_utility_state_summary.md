# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `11`
- rows: `142`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.648717 | 0.716540 | -0.067823 | 26 | 116 | 0.669014 | 0.669014 |
| active/recent utility | 142 | 1.000 | 0.648717 | 0.716540 | -0.067823 | 26 | 116 | 0.669014 | 0.669014 |
| strong utility action | 112 | 0.789 | 0.678740 | 0.762000 | -0.083260 | 9 | 103 | 0.758929 | 0.758929 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 112 | 0.789 | 0.678740 | 0.762000 | -0.083260 | 9 | 103 | 0.758929 | 0.758929 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 142 | 1.000 | 0.648717 | 0.716540 | -0.067823 | 26 | 116 | 0.669014 | 0.669014 |

## Active Smoke/Inferno Intervals

- `9.5s` - `22.5s`, rows `27`
- `28.5s` - `70.5s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.0`, LSTM `0.7231`, XGBoost `0.8791`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.7309`, XGBoost `0.8791`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.7314`, XGBoost `0.8791`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.7333`, XGBoost `0.8807`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.7342`, XGBoost `0.8791`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.7351`, XGBoost `0.8787`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.7189`, XGBoost `0.8597`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.7382`, XGBoost `0.8783`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.7385`, XGBoost `0.8784`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.7410`, XGBoost `0.8793`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
