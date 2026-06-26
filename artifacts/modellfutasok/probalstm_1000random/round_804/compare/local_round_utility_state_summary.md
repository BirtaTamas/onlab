# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `19`
- rows: `251`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 251 | 1.000 | 0.450780 | 0.484801 | -0.034021 | 77 | 174 | 0.374502 | 0.509960 |
| active/recent utility | 251 | 1.000 | 0.450780 | 0.484801 | -0.034021 | 77 | 174 | 0.374502 | 0.509960 |
| strong utility action | 146 | 0.582 | 0.429391 | 0.439756 | -0.010365 | 60 | 86 | 0.267123 | 0.438356 |
| utility damage | 10 | 0.040 | 0.284451 | 0.310718 | -0.026267 | 4 | 6 | 0.000000 | 0.000000 |
| active smoke/inferno | 146 | 0.582 | 0.429391 | 0.439756 | -0.010365 | 60 | 86 | 0.267123 | 0.438356 |
| recent utility last 5s | 10 | 0.040 | 0.508281 | 0.563812 | -0.055531 | 0 | 10 | 0.800000 | 1.000000 |
| flash effect present | 251 | 1.000 | 0.450780 | 0.484801 | -0.034021 | 77 | 174 | 0.374502 | 0.509960 |

## Active Smoke/Inferno Intervals

- `8.0s` - `73.5s`, rows `132`
- `91.0s` - `97.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `96.0`, LSTM `0.2095`, XGBoost `0.3880`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.2175`, XGBoost `0.3880`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.4217`, XGBoost `0.5918`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.4226`, XGBoost `0.5880`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.2291`, XGBoost `0.3880`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.4334`, XGBoost `0.5918`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.4359`, XGBoost `0.5918`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.4393`, XGBoost `0.5876`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4624`, XGBoost `0.3162`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.2423`, XGBoost `0.3880`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
