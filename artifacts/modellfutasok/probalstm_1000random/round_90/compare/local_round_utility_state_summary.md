# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `9`
- rows: `138`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 138 | 1.000 | 0.791677 | 0.786984 | 0.004693 | 57 | 81 | 1.000000 | 0.876812 |
| active/recent utility | 138 | 1.000 | 0.791677 | 0.786984 | 0.004693 | 57 | 81 | 1.000000 | 0.876812 |
| strong utility action | 97 | 0.703 | 0.745516 | 0.735254 | 0.010262 | 50 | 47 | 1.000000 | 0.896907 |
| utility damage | 10 | 0.072 | 0.741688 | 0.802826 | -0.061138 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 85 | 0.616 | 0.766968 | 0.768896 | -0.001927 | 38 | 47 | 1.000000 | 1.000000 |
| recent utility last 5s | 12 | 0.087 | 0.593565 | 0.496961 | 0.096604 | 12 | 0 | 1.000000 | 0.166667 |
| flash effect present | 138 | 1.000 | 0.791677 | 0.786984 | 0.004693 | 57 | 81 | 1.000000 | 0.876812 |

## Active Smoke/Inferno Intervals

- `9.5s` - `51.5s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `5.0`, LSTM `0.6141`, XGBoost `0.4963`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `4.5`, LSTM `0.6045`, XGBoost `0.4899`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `5.5`, LSTM `0.6071`, XGBoost `0.4963`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.6035`, XGBoost `0.4966`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `3.5`, LSTM `0.6021`, XGBoost `0.4966`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `3.0`, LSTM `0.6015`, XGBoost `0.4966`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `2.5`, LSTM `0.5966`, XGBoost `0.4966`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `2.0`, LSTM `0.5901`, XGBoost `0.4966`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `6.0`, LSTM `0.5826`, XGBoost `0.4963`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.5813`, XGBoost `0.4994`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
