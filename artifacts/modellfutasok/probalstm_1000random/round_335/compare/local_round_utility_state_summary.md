# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `4`
- rows: `266`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 266 | 1.000 | 0.512094 | 0.595475 | -0.083381 | 266 | 0 | 0.300752 | 0.236842 |
| active/recent utility | 266 | 1.000 | 0.512094 | 0.595475 | -0.083381 | 266 | 0 | 0.300752 | 0.236842 |
| strong utility action | 167 | 0.628 | 0.670208 | 0.761520 | -0.091311 | 167 | 0 | 0.011976 | 0.000000 |
| utility damage | 10 | 0.038 | 0.549319 | 0.697765 | -0.148446 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 167 | 0.628 | 0.670208 | 0.761520 | -0.091311 | 167 | 0 | 0.011976 | 0.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 266 | 1.000 | 0.512094 | 0.595475 | -0.083381 | 266 | 0 | 0.300752 | 0.236842 |

## Active Smoke/Inferno Intervals

- `8.0s` - `91.0s`, rows `167`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.0`, LSTM `0.5908`, XGBoost `0.8100`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.5917`, XGBoost `0.8100`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.5945`, XGBoost `0.8100`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.5967`, XGBoost `0.8100`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.5972`, XGBoost `0.8100`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.5988`, XGBoost `0.8100`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.6008`, XGBoost `0.8100`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.6027`, XGBoost `0.8100`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.6087`, XGBoost `0.8100`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.6121`, XGBoost `0.8028`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
