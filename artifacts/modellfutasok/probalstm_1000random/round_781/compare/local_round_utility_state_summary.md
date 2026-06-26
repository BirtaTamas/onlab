# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `9`
- rows: `124`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 124 | 1.000 | 0.057902 | 0.146670 | -0.088769 | 124 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 124 | 1.000 | 0.057902 | 0.146670 | -0.088769 | 124 | 0 | 1.000000 | 1.000000 |
| strong utility action | 94 | 0.758 | 0.063226 | 0.165972 | -0.102745 | 94 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 94 | 0.758 | 0.063226 | 0.165972 | -0.102745 | 94 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 124 | 1.000 | 0.057902 | 0.146670 | -0.088769 | 124 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `4.5s` - `44.0s`, rows `80`
- `45.5s` - `52.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `7.0`, LSTM `0.1092`, XGBoost `0.2746`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0718`, XGBoost `0.2360`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.0323`, XGBoost `0.1957`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0938`, XGBoost `0.2566`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0331`, XGBoost `0.1957`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.1129`, XGBoost `0.2746`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0936`, XGBoost `0.2538`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.1150`, XGBoost `0.2746`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0959`, XGBoost `0.2549`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.0910`, XGBoost `0.2482`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
