# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `4`
- rows: `292`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 292 | 1.000 | 0.349331 | 0.358710 | -0.009379 | 195 | 97 | 0.455479 | 0.472603 |
| active/recent utility | 292 | 1.000 | 0.349331 | 0.358710 | -0.009379 | 195 | 97 | 0.455479 | 0.472603 |
| strong utility action | 177 | 0.606 | 0.511404 | 0.516717 | -0.005313 | 89 | 88 | 0.209040 | 0.237288 |
| utility damage | 31 | 0.106 | 0.530993 | 0.553556 | -0.022562 | 20 | 11 | 0.193548 | 0.193548 |
| active smoke/inferno | 177 | 0.606 | 0.511404 | 0.516717 | -0.005313 | 89 | 88 | 0.209040 | 0.237288 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 292 | 1.000 | 0.349331 | 0.358710 | -0.009379 | 195 | 97 | 0.455479 | 0.472603 |

## Active Smoke/Inferno Intervals

- `8.5s` - `62.5s`, rows `109`
- `64.0s` - `97.5s`, rows `68`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.5`, LSTM `0.3174`, XGBoost `0.4233`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `43.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.3179`, XGBoost `0.4217`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `43.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.3221`, XGBoost `0.4236`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `43.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.3284`, XGBoost `0.4212`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `43.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.5789`, XGBoost `0.4939`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.3727`, XGBoost `0.4547`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.6043`, XGBoost `0.5247`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.6122`, XGBoost `0.5349`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.8123`, XGBoost `0.7359`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `16.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.3495`, XGBoost `0.4256`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
