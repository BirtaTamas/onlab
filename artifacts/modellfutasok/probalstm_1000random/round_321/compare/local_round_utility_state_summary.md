# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv`
- round_num: `11`
- rows: `210`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 210 | 1.000 | 0.584568 | 0.542063 | 0.042505 | 159 | 51 | 0.757143 | 0.690476 |
| active/recent utility | 210 | 1.000 | 0.584568 | 0.542063 | 0.042505 | 159 | 51 | 0.757143 | 0.690476 |
| strong utility action | 176 | 0.838 | 0.598639 | 0.558189 | 0.040450 | 127 | 49 | 0.784091 | 0.721591 |
| utility damage | 15 | 0.071 | 0.554301 | 0.475363 | 0.078938 | 12 | 3 | 0.533333 | 0.533333 |
| active smoke/inferno | 176 | 0.838 | 0.598639 | 0.558189 | 0.040450 | 127 | 49 | 0.784091 | 0.721591 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 210 | 1.000 | 0.584568 | 0.542063 | 0.042505 | 159 | 51 | 0.757143 | 0.690476 |

## Active Smoke/Inferno Intervals

- `9.0s` - `79.5s`, rows `142`
- `88.0s` - `104.5s`, rows `34`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.0`, LSTM `0.1688`, XGBoost `0.3714`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.1730`, XGBoost `0.3714`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.1694`, XGBoost `0.3658`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.1802`, XGBoost `0.3629`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.5230`, XGBoost `0.3424`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.2004`, XGBoost `0.3790`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.7553`, XGBoost `0.5784`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.7541`, XGBoost `0.5784`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.5167`, XGBoost `0.3424`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.5158`, XGBoost `0.3424`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
