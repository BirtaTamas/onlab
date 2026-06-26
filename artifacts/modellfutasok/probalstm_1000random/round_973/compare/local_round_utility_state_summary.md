# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-flyquest-bo3-ElcEZT56lTCLJYDcWlMY2d/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `11`
- rows: `162`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 162 | 1.000 | 0.517130 | 0.620536 | -0.103406 | 0 | 162 | 0.654321 | 1.000000 |
| active/recent utility | 162 | 1.000 | 0.517130 | 0.620536 | -0.103406 | 0 | 162 | 0.654321 | 1.000000 |
| strong utility action | 146 | 0.901 | 0.520850 | 0.626436 | -0.105586 | 0 | 146 | 0.698630 | 1.000000 |
| utility damage | 23 | 0.142 | 0.497776 | 0.606892 | -0.109116 | 0 | 23 | 0.478261 | 1.000000 |
| active smoke/inferno | 146 | 0.901 | 0.520850 | 0.626436 | -0.105586 | 0 | 146 | 0.698630 | 1.000000 |
| recent utility last 5s | 10 | 0.062 | 0.542458 | 0.629196 | -0.086738 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 162 | 1.000 | 0.517130 | 0.620536 | -0.103406 | 0 | 162 | 0.654321 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `42.0s`, rows `73`
- `44.5s` - `80.5s`, rows `73`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `74.0`, LSTM `0.3132`, XGBoost `0.5423`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.3518`, XGBoost `0.5574`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.3645`, XGBoost `0.5455`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.3777`, XGBoost `0.5566`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.3831`, XGBoost `0.5581`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.3657`, XGBoost `0.5375`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.4602`, XGBoost `0.6254`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4007`, XGBoost `0.5558`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.4122`, XGBoost `0.5662`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.4046`, XGBoost `0.5581`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
