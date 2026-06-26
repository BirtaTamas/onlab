# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-falcons-vs-mouz-bo3-plkh_Ps38mI3o_rFlgAljz/falcons-vs-mouz-m3-nuke-p3.csv`
- round_num: `1`
- rows: `184`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 184 | 1.000 | 0.213423 | 0.310225 | -0.096803 | 153 | 31 | 1.000000 | 0.940217 |
| active/recent utility | 184 | 1.000 | 0.213423 | 0.310225 | -0.096803 | 153 | 31 | 1.000000 | 0.940217 |
| strong utility action | 138 | 0.750 | 0.239874 | 0.308085 | -0.068211 | 107 | 31 | 1.000000 | 0.942029 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 138 | 0.750 | 0.239874 | 0.308085 | -0.068211 | 107 | 31 | 1.000000 | 0.942029 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 184 | 1.000 | 0.213423 | 0.310225 | -0.096803 | 153 | 31 | 1.000000 | 0.940217 |

## Active Smoke/Inferno Intervals

- `11.0s` - `67.5s`, rows `114`
- `71.0s` - `78.0s`, rows `15`
- `84.0s` - `86.0s`, rows `5`
- `90.0s` - `91.5s`, rows `4`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.0`, LSTM `0.0572`, XGBoost `0.5016`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `76.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.0909`, XGBoost `0.5244`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `75.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.1759`, XGBoost `0.5920`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.0401`, XGBoost `0.4489`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `76.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.1633`, XGBoost `0.5698`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.1188`, XGBoost `0.5244`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `69.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.1448`, XGBoost `0.5328`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.1980`, XGBoost `0.5705`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `29.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.2100`, XGBoost `0.5821`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.0461`, XGBoost `0.3510`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `40.0`, recent_utility `0`
