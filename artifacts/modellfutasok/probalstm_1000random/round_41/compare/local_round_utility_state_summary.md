# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-liquid-vs-furia-bo3-oYHD2J45okzf6eapD2F9CM/liquid-vs-furia-m1-mirage.csv`
- round_num: `8`
- rows: `128`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 128 | 1.000 | 0.830486 | 0.807453 | 0.023033 | 75 | 53 | 1.000000 | 0.984375 |
| active/recent utility | 128 | 1.000 | 0.830486 | 0.807453 | 0.023033 | 75 | 53 | 1.000000 | 0.984375 |
| strong utility action | 125 | 0.977 | 0.836367 | 0.814610 | 0.021757 | 72 | 53 | 1.000000 | 0.984000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 115 | 0.898 | 0.858327 | 0.841342 | 0.016985 | 62 | 53 | 1.000000 | 0.982609 |
| recent utility last 5s | 10 | 0.078 | 0.583828 | 0.507190 | 0.076638 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 128 | 1.000 | 0.830486 | 0.807453 | 0.023033 | 75 | 53 | 1.000000 | 0.984375 |

## Active Smoke/Inferno Intervals

- `6.5s` - `63.5s`, rows `115`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `16.0`, LSTM `0.7689`, XGBoost `0.6641`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `3.0`, LSTM `0.6041`, XGBoost `0.5048`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `15.0`, LSTM `0.7667`, XGBoost `0.6675`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.5990`, XGBoost `0.5022`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `1.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6012`, XGBoost `0.5057`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `2.5`, LSTM `0.6079`, XGBoost `0.5127`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.5996`, XGBoost `0.5048`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `11.5`, LSTM `0.6012`, XGBoost `0.5077`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `1.0`, recent_utility `0`
- seconds `4.0`, LSTM `0.5983`, XGBoost `0.5048`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `10.5`, LSTM `0.5994`, XGBoost `0.5070`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
