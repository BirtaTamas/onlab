# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `15`
- rows: `143`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 143 | 1.000 | 0.136238 | 0.210296 | -0.074058 | 131 | 12 | 1.000000 | 1.000000 |
| active/recent utility | 143 | 1.000 | 0.136238 | 0.210296 | -0.074058 | 131 | 12 | 1.000000 | 1.000000 |
| strong utility action | 116 | 0.811 | 0.136593 | 0.211498 | -0.074906 | 105 | 11 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 105 | 0.734 | 0.146137 | 0.215749 | -0.069613 | 94 | 11 | 1.000000 | 1.000000 |
| recent utility last 5s | 11 | 0.077 | 0.045490 | 0.170920 | -0.125430 | 11 | 0 | 1.000000 | 1.000000 |
| flash effect present | 143 | 1.000 | 0.136238 | 0.210296 | -0.074058 | 131 | 12 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `30.0s`, rows `44`
- `36.0s` - `42.5s`, rows `14`
- `46.5s` - `69.5s`, rows `47`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.5`, LSTM `0.0339`, XGBoost `0.1760`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.5`, LSTM `0.0439`, XGBoost `0.1823`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.0416`, XGBoost `0.1791`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `6.0`, LSTM `0.0446`, XGBoost `0.1818`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `5.5`, LSTM `0.0470`, XGBoost `0.1798`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `11.0`, LSTM `0.0547`, XGBoost `0.1851`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0559`, XGBoost `0.1851`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0567`, XGBoost `0.1851`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0610`, XGBoost `0.1874`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.0703`, XGBoost `0.1953`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `32.0`, recent_utility `0`
