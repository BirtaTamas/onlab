# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-og-inferno-UyQlNJx_rptvvsTtINI5j3/virtus-pro-vs-og-inferno.csv`
- round_num: `3`
- rows: `223`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 223 | 1.000 | 0.561730 | 0.532709 | 0.029021 | 164 | 59 | 0.995516 | 0.843049 |
| active/recent utility | 223 | 1.000 | 0.561730 | 0.532709 | 0.029021 | 164 | 59 | 0.995516 | 0.843049 |
| strong utility action | 158 | 0.709 | 0.544581 | 0.515164 | 0.029417 | 131 | 27 | 0.993671 | 0.791139 |
| utility damage | 19 | 0.085 | 0.540153 | 0.497694 | 0.042459 | 17 | 2 | 0.947368 | 0.315789 |
| active smoke/inferno | 156 | 0.700 | 0.543209 | 0.514759 | 0.028450 | 129 | 27 | 0.993590 | 0.794872 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 223 | 1.000 | 0.561730 | 0.532709 | 0.029021 | 164 | 59 | 0.995516 | 0.843049 |

## Active Smoke/Inferno Intervals

- `9.5s` - `44.5s`, rows `71`
- `61.0s` - `103.0s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `103.5`, LSTM `0.7965`, XGBoost `0.6709`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `51.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.5058`, XGBoost `0.3915`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `51.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.6335`, XGBoost `0.5206`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6386`, XGBoost `0.5269`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.6326`, XGBoost `0.5269`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6241`, XGBoost `0.5206`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.6282`, XGBoost `0.5269`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6255`, XGBoost `0.5267`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6211`, XGBoost `0.5250`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.6209`, XGBoost `0.5257`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
