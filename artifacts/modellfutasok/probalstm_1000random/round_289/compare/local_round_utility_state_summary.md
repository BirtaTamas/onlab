# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `3`
- rows: `305`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 305 | 1.000 | 0.093621 | 0.117637 | -0.024016 | 256 | 49 | 1.000000 | 1.000000 |
| active/recent utility | 305 | 1.000 | 0.093621 | 0.117637 | -0.024016 | 256 | 49 | 1.000000 | 1.000000 |
| strong utility action | 152 | 0.498 | 0.161829 | 0.187592 | -0.025763 | 115 | 37 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 152 | 0.498 | 0.161829 | 0.187592 | -0.025763 | 115 | 37 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 305 | 1.000 | 0.093621 | 0.117637 | -0.024016 | 256 | 49 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `32.5s`, rows `49`
- `35.0s` - `86.0s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.0`, LSTM `0.0971`, XGBoost `0.2495`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0793`, XGBoost `0.2231`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.1137`, XGBoost `0.2549`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1165`, XGBoost `0.2562`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.1152`, XGBoost `0.2540`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0994`, XGBoost `0.2129`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `35.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1018`, XGBoost `0.2142`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `35.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1020`, XGBoost `0.2141`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `35.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1028`, XGBoost `0.2137`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `35.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.1072`, XGBoost `0.2164`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `35.0`, recent_utility `0`
