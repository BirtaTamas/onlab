# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-furia-vs-g2-bo3-QMek4tXQesgbTlulfGKOmD/furia-vs-g2-m1-inferno.csv`
- round_num: `1`
- rows: `135`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.467510 | 0.553369 | -0.085859 | 124 | 11 | 0.303704 | 0.192593 |
| active/recent utility | 114 | 0.844 | 0.460824 | 0.560172 | -0.099347 | 103 | 11 | 0.333333 | 0.228070 |
| strong utility action | 88 | 0.652 | 0.529097 | 0.648086 | -0.118990 | 85 | 3 | 0.159091 | 0.068182 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 88 | 0.652 | 0.529097 | 0.648086 | -0.118990 | 85 | 3 | 0.159091 | 0.068182 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 114 | 0.844 | 0.460824 | 0.560172 | -0.099347 | 103 | 11 | 0.333333 | 0.228070 |

## Active Smoke/Inferno Intervals

- `11.0s` - `32.5s`, rows `44`
- `35.5s` - `57.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.0`, LSTM `0.5735`, XGBoost `0.8113`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.5795`, XGBoost `0.8145`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5301`, XGBoost `0.7268`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5291`, XGBoost `0.7235`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5167`, XGBoost `0.7110`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5314`, XGBoost `0.7256`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5264`, XGBoost `0.7193`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.5319`, XGBoost `0.7195`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.4339`, XGBoost `0.6188`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5411`, XGBoost `0.7254`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
