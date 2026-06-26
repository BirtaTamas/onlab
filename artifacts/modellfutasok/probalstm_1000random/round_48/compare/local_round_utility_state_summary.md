# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m3-inferno.csv`
- round_num: `12`
- rows: `205`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 205 | 1.000 | 0.753874 | 0.716114 | 0.037759 | 146 | 59 | 1.000000 | 1.000000 |
| active/recent utility | 205 | 1.000 | 0.753874 | 0.716114 | 0.037759 | 146 | 59 | 1.000000 | 1.000000 |
| strong utility action | 173 | 0.844 | 0.753525 | 0.716086 | 0.037439 | 124 | 49 | 1.000000 | 1.000000 |
| utility damage | 25 | 0.122 | 0.817976 | 0.768934 | 0.049042 | 15 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 157 | 0.766 | 0.765227 | 0.729480 | 0.035747 | 108 | 49 | 1.000000 | 1.000000 |
| recent utility last 5s | 30 | 0.146 | 0.741883 | 0.721295 | 0.020588 | 20 | 10 | 1.000000 | 1.000000 |
| flash effect present | 205 | 1.000 | 0.753874 | 0.716114 | 0.037759 | 146 | 59 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `52.5s`, rows `84`
- `61.0s` - `97.0s`, rows `73`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.0`, LSTM `0.7027`, XGBoost `0.5648`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.6814`, XGBoost `0.5547`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.7352`, XGBoost `0.6174`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `66.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.7349`, XGBoost `0.6174`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `66.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.6765`, XGBoost `0.5594`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.7305`, XGBoost `0.6174`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `66.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.6719`, XGBoost `0.5594`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6807`, XGBoost `0.5686`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.7286`, XGBoost `0.6174`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `66.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.7290`, XGBoost `0.6181`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `65.0`, recent_utility `0`
