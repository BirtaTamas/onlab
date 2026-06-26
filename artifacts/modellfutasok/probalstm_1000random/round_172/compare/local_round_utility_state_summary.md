# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `15`
- rows: `166`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 166 | 1.000 | 0.008673 | 0.024104 | -0.015431 | 166 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 166 | 1.000 | 0.008673 | 0.024104 | -0.015431 | 166 | 0 | 1.000000 | 1.000000 |
| strong utility action | 124 | 0.747 | 0.009161 | 0.021829 | -0.012669 | 124 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 124 | 0.747 | 0.009161 | 0.021829 | -0.012669 | 124 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 166 | 1.000 | 0.008673 | 0.024104 | -0.015431 | 166 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `68.5s`, rows `124`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.0`, LSTM `0.0187`, XGBoost `0.0878`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0190`, XGBoost `0.0879`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0197`, XGBoost `0.0873`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0197`, XGBoost `0.0870`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0221`, XGBoost `0.0879`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.0199`, XGBoost `0.0844`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.0201`, XGBoost `0.0830`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0185`, XGBoost `0.0744`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `4.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0080`, XGBoost `0.0433`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0090`, XGBoost `0.0415`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `12.0`, recent_utility `0`
