# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-astralis-bo3-Zley6FZuKcttfrliAqsvWJ/astralis-vs-vitality-m1-inferno.csv`
- round_num: `13`
- rows: `113`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 113 | 1.000 | 0.249848 | 0.324418 | -0.074570 | 81 | 32 | 0.716814 | 0.734513 |
| active/recent utility | 113 | 1.000 | 0.249848 | 0.324418 | -0.074570 | 81 | 32 | 0.716814 | 0.734513 |
| strong utility action | 44 | 0.389 | 0.427203 | 0.451722 | -0.024519 | 22 | 22 | 0.568182 | 0.613636 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.389 | 0.427203 | 0.451722 | -0.024519 | 22 | 22 | 0.568182 | 0.613636 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 113 | 1.000 | 0.249848 | 0.324418 | -0.074570 | 81 | 32 | 0.716814 | 0.734513 |

## Active Smoke/Inferno Intervals

- `6.5s` - `28.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.5`, LSTM `0.2709`, XGBoost `0.4830`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.2782`, XGBoost `0.4830`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.2768`, XGBoost `0.4770`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.3760`, XGBoost `0.1975`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.3054`, XGBoost `0.4771`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.3073`, XGBoost `0.4718`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.3179`, XGBoost `0.4684`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.3412`, XGBoost `0.4730`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5450`, XGBoost `0.4435`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5234`, XGBoost `0.4306`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
