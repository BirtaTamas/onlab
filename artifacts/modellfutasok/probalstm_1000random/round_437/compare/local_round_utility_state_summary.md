# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `7`
- rows: `264`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 264 | 1.000 | 0.098789 | 0.178305 | -0.079516 | 263 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 264 | 1.000 | 0.098789 | 0.178305 | -0.079516 | 263 | 1 | 1.000000 | 1.000000 |
| strong utility action | 203 | 0.769 | 0.126198 | 0.227783 | -0.101584 | 202 | 1 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 191 | 0.723 | 0.120486 | 0.223047 | -0.102561 | 190 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 12 | 0.045 | 0.217125 | 0.303161 | -0.086036 | 12 | 0 | 1.000000 | 1.000000 |
| flash effect present | 264 | 1.000 | 0.098789 | 0.178305 | -0.079516 | 263 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `97.0s`, rows `177`
- `102.5s` - `109.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.0`, LSTM `0.1162`, XGBoost `0.2971`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.1181`, XGBoost `0.2967`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1090`, XGBoost `0.2862`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.1315`, XGBoost `0.3078`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1135`, XGBoost `0.2879`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1406`, XGBoost `0.3138`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1351`, XGBoost `0.3078`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.1487`, XGBoost `0.3172`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.1260`, XGBoost `0.2938`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.1161`, XGBoost `0.2837`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `32.0`, recent_utility `0`
