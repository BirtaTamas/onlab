# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `4`
- rows: `157`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.247752 | 0.241265 | 0.006487 | 89 | 68 | 0.840764 | 0.891720 |
| active/recent utility | 157 | 1.000 | 0.247752 | 0.241265 | 0.006487 | 89 | 68 | 0.840764 | 0.891720 |
| strong utility action | 134 | 0.854 | 0.235288 | 0.227773 | 0.007515 | 79 | 55 | 0.813433 | 0.873134 |
| utility damage | 12 | 0.076 | 0.612716 | 0.575356 | 0.037360 | 4 | 8 | 0.000000 | 0.000000 |
| active smoke/inferno | 134 | 0.854 | 0.235288 | 0.227773 | 0.007515 | 79 | 55 | 0.813433 | 0.873134 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 157 | 1.000 | 0.247752 | 0.241265 | 0.006487 | 89 | 68 | 0.840764 | 0.891720 |

## Active Smoke/Inferno Intervals

- `9.5s` - `14.5s`, rows `11`
- `15.5s` - `73.0s`, rows `116`
- `75.0s` - `78.0s`, rows `7`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.0`, LSTM `0.4657`, XGBoost `0.2911`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.4572`, XGBoost `0.2841`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.4506`, XGBoost `0.2801`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4542`, XGBoost `0.2892`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.4931`, XGBoost `0.3359`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `7.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.4524`, XGBoost `0.2988`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.4881`, XGBoost `0.3359`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `7.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.4481`, XGBoost `0.2979`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.3217`, XGBoost `0.1723`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.0255`, XGBoost `0.1715`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
