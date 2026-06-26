# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `15`
- rows: `154`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 154 | 1.000 | 0.821781 | 0.811237 | 0.010544 | 67 | 87 | 1.000000 | 1.000000 |
| active/recent utility | 154 | 1.000 | 0.821781 | 0.811237 | 0.010544 | 67 | 87 | 1.000000 | 1.000000 |
| strong utility action | 122 | 0.792 | 0.836378 | 0.831552 | 0.004825 | 48 | 74 | 1.000000 | 1.000000 |
| utility damage | 21 | 0.136 | 0.797959 | 0.769134 | 0.028826 | 11 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 122 | 0.792 | 0.836378 | 0.831552 | 0.004825 | 48 | 74 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.065 | 0.890829 | 0.836644 | 0.054185 | 9 | 1 | 1.000000 | 1.000000 |
| flash effect present | 154 | 1.000 | 0.821781 | 0.811237 | 0.010544 | 67 | 87 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `64.5s`, rows `111`
- `66.5s` - `71.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.0`, LSTM `0.8800`, XGBoost `0.7852`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `10.0`, LSTM `0.6710`, XGBoost `0.5847`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6663`, XGBoost `0.5840`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.8626`, XGBoost `0.7819`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `62.0`, LSTM `0.8613`, XGBoost `0.7835`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `11.0`, LSTM `0.6687`, XGBoost `0.5919`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `22.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.8611`, XGBoost `0.7847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `9.5`, LSTM `0.6609`, XGBoost `0.5847`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.8643`, XGBoost `0.7919`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `60.0`, LSTM `0.8668`, XGBoost `0.7948`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
