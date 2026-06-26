# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `3`
- rows: `288`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 288 | 1.000 | 0.094193 | 0.126453 | -0.032260 | 227 | 61 | 1.000000 | 1.000000 |
| active/recent utility | 288 | 1.000 | 0.094193 | 0.126453 | -0.032260 | 227 | 61 | 1.000000 | 1.000000 |
| strong utility action | 168 | 0.583 | 0.129424 | 0.170589 | -0.041166 | 146 | 22 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 168 | 0.583 | 0.129424 | 0.170589 | -0.041166 | 146 | 22 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 288 | 1.000 | 0.094193 | 0.126453 | -0.032260 | 227 | 61 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `69.0s`, rows `121`
- `81.0s` - `104.0s`, rows `47`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.0`, LSTM `0.0738`, XGBoost `0.2088`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.0859`, XGBoost `0.2146`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0894`, XGBoost `0.2088`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0936`, XGBoost `0.2099`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.1000`, XGBoost `0.2146`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0980`, XGBoost `0.2099`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.1016`, XGBoost `0.2099`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.0432`, XGBoost `0.1480`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.0513`, XGBoost `0.1556`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.0447`, XGBoost `0.1489`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
