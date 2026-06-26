# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m3-ancient.csv`
- round_num: `18`
- rows: `154`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 154 | 1.000 | 0.092353 | 0.180326 | -0.087973 | 143 | 11 | 1.000000 | 1.000000 |
| active/recent utility | 154 | 1.000 | 0.092353 | 0.180326 | -0.087973 | 143 | 11 | 1.000000 | 1.000000 |
| strong utility action | 113 | 0.734 | 0.097287 | 0.182150 | -0.084862 | 103 | 10 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 113 | 0.734 | 0.097287 | 0.182150 | -0.084862 | 103 | 10 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.065 | 0.312142 | 0.317630 | -0.005488 | 7 | 3 | 1.000000 | 1.000000 |
| flash effect present | 154 | 1.000 | 0.092353 | 0.180326 | -0.087973 | 143 | 11 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `55.5s`, rows `99`
- `64.0s` - `70.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `55.5`, LSTM `0.0233`, XGBoost `0.1829`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.0274`, XGBoost `0.1829`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.0219`, XGBoost `0.1750`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.0282`, XGBoost `0.1791`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0251`, XGBoost `0.1754`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.0323`, XGBoost `0.1814`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.0248`, XGBoost `0.1722`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0296`, XGBoost `0.1750`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0417`, XGBoost `0.1869`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.0378`, XGBoost `0.1829`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
