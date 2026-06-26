# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `13`
- rows: `228`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 228 | 1.000 | 0.546436 | 0.574323 | -0.027887 | 177 | 51 | 0.258772 | 0.258772 |
| active/recent utility | 228 | 1.000 | 0.546436 | 0.574323 | -0.027887 | 177 | 51 | 0.258772 | 0.258772 |
| strong utility action | 167 | 0.732 | 0.643214 | 0.677300 | -0.034086 | 136 | 31 | 0.125749 | 0.125749 |
| utility damage | 10 | 0.044 | 0.812214 | 0.853388 | -0.041174 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 167 | 0.732 | 0.643214 | 0.677300 | -0.034086 | 136 | 31 | 0.125749 | 0.125749 |
| recent utility last 5s | 10 | 0.044 | 0.793945 | 0.831789 | -0.037844 | 10 | 0 | 0.000000 | 0.000000 |
| flash effect present | 228 | 1.000 | 0.546436 | 0.574323 | -0.027887 | 177 | 51 | 0.258772 | 0.258772 |

## Active Smoke/Inferno Intervals

- `11.5s` - `94.5s`, rows `167`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `84.5`, LSTM `0.3473`, XGBoost `0.1479`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.7189`, XGBoost `0.8405`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.7217`, XGBoost `0.8405`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.7222`, XGBoost `0.8405`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.0143`, XGBoost `0.1273`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.7311`, XGBoost `0.8392`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.6838`, XGBoost `0.7907`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.7480`, XGBoost `0.8395`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.0112`, XGBoost `0.1019`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.0130`, XGBoost `0.1037`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
