# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `15`
- rows: `222`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 222 | 1.000 | 0.395084 | 0.322998 | 0.072086 | 135 | 87 | 0.603604 | 0.265766 |
| active/recent utility | 222 | 1.000 | 0.395084 | 0.322998 | 0.072086 | 135 | 87 | 0.603604 | 0.265766 |
| strong utility action | 148 | 0.667 | 0.459252 | 0.344001 | 0.115251 | 120 | 28 | 0.756757 | 0.250000 |
| utility damage | 10 | 0.045 | 0.569870 | 0.439005 | 0.130865 | 10 | 0 | 1.000000 | 0.500000 |
| active smoke/inferno | 138 | 0.622 | 0.448434 | 0.329795 | 0.118639 | 110 | 28 | 0.739130 | 0.195652 |
| recent utility last 5s | 10 | 0.045 | 0.608542 | 0.540042 | 0.068500 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 222 | 1.000 | 0.395084 | 0.322998 | 0.072086 | 135 | 87 | 0.603604 | 0.265766 |

## Active Smoke/Inferno Intervals

- `8.5s` - `77.0s`, rows `138`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.0`, LSTM `0.4217`, XGBoost `0.1284`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.3834`, XGBoost `0.1281`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.3632`, XGBoost `0.1338`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.5355`, XGBoost `0.3203`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5304`, XGBoost `0.3203`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.5470`, XGBoost `0.3389`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5267`, XGBoost `0.3203`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5260`, XGBoost `0.3230`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `13.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5249`, XGBoost `0.3230`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `13.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5238`, XGBoost `0.3230`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `13.0`, recent_utility `0`
