# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `7`
- rows: `268`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 268 | 1.000 | 0.372960 | 0.401414 | -0.028454 | 172 | 96 | 0.679104 | 0.835821 |
| active/recent utility | 268 | 1.000 | 0.372960 | 0.401414 | -0.028454 | 172 | 96 | 0.679104 | 0.835821 |
| strong utility action | 218 | 0.813 | 0.414410 | 0.446719 | -0.032309 | 127 | 91 | 0.605505 | 0.798165 |
| utility damage | 10 | 0.037 | 0.477790 | 0.495400 | -0.017610 | 10 | 0 | 1.000000 | 0.900000 |
| active smoke/inferno | 218 | 0.813 | 0.414410 | 0.446719 | -0.032309 | 127 | 91 | 0.605505 | 0.798165 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 268 | 1.000 | 0.372960 | 0.401414 | -0.028454 | 172 | 96 | 0.679104 | 0.835821 |

## Active Smoke/Inferno Intervals

- `9.0s` - `34.5s`, rows `52`
- `36.0s` - `118.5s`, rows `166`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `104.5`, LSTM `0.1038`, XGBoost `0.4224`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.1020`, XGBoost `0.4136`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.1246`, XGBoost `0.4254`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.0`, LSTM `0.0694`, XGBoost `0.3667`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.0689`, XGBoost `0.3526`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.5`, LSTM `0.0790`, XGBoost `0.3554`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.5`, LSTM `0.0696`, XGBoost `0.3444`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.0699`, XGBoost `0.3415`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.0965`, XGBoost `0.3661`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.1206`, XGBoost `0.3873`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
