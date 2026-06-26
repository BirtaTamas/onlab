# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `1`
- rows: `139`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 139 | 1.000 | 0.334313 | 0.408007 | -0.073695 | 136 | 3 | 0.697842 | 0.618705 |
| active/recent utility | 116 | 0.835 | 0.299257 | 0.380687 | -0.081430 | 113 | 3 | 0.836207 | 0.741379 |
| strong utility action | 85 | 0.612 | 0.343131 | 0.419816 | -0.076685 | 84 | 1 | 0.823529 | 0.705882 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 85 | 0.612 | 0.343131 | 0.419816 | -0.076685 | 84 | 1 | 0.823529 | 0.705882 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 45 | 0.324 | 0.177053 | 0.261926 | -0.084873 | 43 | 2 | 0.911111 | 0.888889 |

## Active Smoke/Inferno Intervals

- `11.5s` - `53.5s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.5`, LSTM `0.3283`, XGBoost `0.5331`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.3426`, XGBoost `0.5338`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.3475`, XGBoost `0.5253`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.2928`, XGBoost `0.4698`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.2936`, XGBoost `0.4698`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.3051`, XGBoost `0.4698`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.3723`, XGBoost `0.5353`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.3399`, XGBoost `0.4937`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.3818`, XGBoost `0.5353`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.3223`, XGBoost `0.4698`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
