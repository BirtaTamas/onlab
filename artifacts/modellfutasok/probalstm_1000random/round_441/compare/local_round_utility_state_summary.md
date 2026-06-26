# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv`
- round_num: `4`
- rows: `141`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 141 | 1.000 | 0.837194 | 0.823465 | 0.013729 | 59 | 82 | 1.000000 | 0.971631 |
| active/recent utility | 141 | 1.000 | 0.837194 | 0.823465 | 0.013729 | 59 | 82 | 1.000000 | 0.971631 |
| strong utility action | 125 | 0.887 | 0.853212 | 0.839204 | 0.014008 | 50 | 75 | 1.000000 | 0.968000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 125 | 0.887 | 0.853212 | 0.839204 | 0.014008 | 50 | 75 | 1.000000 | 0.968000 |
| recent utility last 5s | 10 | 0.071 | 0.697075 | 0.703401 | -0.006326 | 4 | 6 | 1.000000 | 1.000000 |
| flash effect present | 141 | 1.000 | 0.837194 | 0.823465 | 0.013729 | 59 | 82 | 1.000000 | 0.971631 |

## Active Smoke/Inferno Intervals

- `8.0s` - `70.0s`, rows `125`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.5`, LSTM `0.6439`, XGBoost `0.4901`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6381`, XGBoost `0.4901`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.6215`, XGBoost `0.4901`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.7881`, XGBoost `0.6680`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.6075`, XGBoost `0.4894`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.7807`, XGBoost `0.6662`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.7791`, XGBoost `0.6680`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.7725`, XGBoost `0.6662`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.7731`, XGBoost `0.6680`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.7797`, XGBoost `0.6746`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
