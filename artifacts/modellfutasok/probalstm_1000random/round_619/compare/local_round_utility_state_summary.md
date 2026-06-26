# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `10`
- rows: `184`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 184 | 1.000 | 0.709940 | 0.734327 | -0.024387 | 158 | 26 | 0.027174 | 0.000000 |
| active/recent utility | 184 | 1.000 | 0.709940 | 0.734327 | -0.024387 | 158 | 26 | 0.027174 | 0.000000 |
| strong utility action | 102 | 0.554 | 0.628458 | 0.657852 | -0.029394 | 94 | 8 | 0.049020 | 0.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 102 | 0.554 | 0.628458 | 0.657852 | -0.029394 | 94 | 8 | 0.049020 | 0.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 184 | 1.000 | 0.709940 | 0.734327 | -0.024387 | 158 | 26 | 0.027174 | 0.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `58.5s`, rows `102`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `45.0`, LSTM `0.6337`, XGBoost `0.7518`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.6613`, XGBoost `0.7478`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4928`, XGBoost `0.5544`, closer `lstm`, smoke `2`, inferno `5`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.7059`, XGBoost `0.7648`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4959`, XGBoost `0.5544`, closer `lstm`, smoke `2`, inferno `5`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.4948`, XGBoost `0.5531`, closer `lstm`, smoke `2`, inferno `5`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.6866`, XGBoost `0.7448`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `8.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.6868`, XGBoost `0.7448`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.4959`, XGBoost `0.5531`, closer `lstm`, smoke `3`, inferno `6`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.4962`, XGBoost `0.5517`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
