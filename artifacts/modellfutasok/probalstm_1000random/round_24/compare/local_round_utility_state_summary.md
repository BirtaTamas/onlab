# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `21`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.869511 | 0.898332 | -0.028821 | 21 | 209 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.869511 | 0.898332 | -0.028821 | 21 | 209 | 1.000000 | 1.000000 |
| strong utility action | 128 | 0.557 | 0.861398 | 0.911559 | -0.050161 | 2 | 126 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.043 | 0.655305 | 0.702242 | -0.046937 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 128 | 0.557 | 0.861398 | 0.911559 | -0.050161 | 2 | 126 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.869511 | 0.898332 | -0.028821 | 21 | 209 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `72.0s`, rows `128`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `17.5`, LSTM `0.7891`, XGBoost `0.9148`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.7848`, XGBoost `0.9096`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.8113`, XGBoost `0.9351`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `20.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.7913`, XGBoost `0.9096`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `16.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.8441`, XGBoost `0.9384`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.8419`, XGBoost `0.9351`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.8464`, XGBoost `0.9381`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.8519`, XGBoost `0.9373`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.6473`, XGBoost `0.7304`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.8556`, XGBoost `0.9373`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
