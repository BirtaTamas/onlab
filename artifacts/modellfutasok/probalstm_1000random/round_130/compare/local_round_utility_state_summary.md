# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-rare-atom-vs-astralis-bo3-2mbRF781jI0kkV-FX6ZCr7/rare-atom-vs-astralis-m1-ancient.csv`
- round_num: `9`
- rows: `156`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 156 | 1.000 | 0.772741 | 0.761912 | 0.010829 | 69 | 87 | 1.000000 | 0.948718 |
| active/recent utility | 156 | 1.000 | 0.772741 | 0.761912 | 0.010829 | 69 | 87 | 1.000000 | 0.948718 |
| strong utility action | 116 | 0.744 | 0.713544 | 0.692094 | 0.021450 | 66 | 50 | 1.000000 | 0.931034 |
| utility damage | 28 | 0.179 | 0.746303 | 0.725498 | 0.020805 | 17 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 106 | 0.679 | 0.719235 | 0.706128 | 0.013107 | 56 | 50 | 1.000000 | 0.924528 |
| recent utility last 5s | 20 | 0.128 | 0.619967 | 0.518759 | 0.101208 | 19 | 1 | 1.000000 | 0.600000 |
| flash effect present | 156 | 1.000 | 0.772741 | 0.761912 | 0.010829 | 69 | 87 | 1.000000 | 0.948718 |

## Active Smoke/Inferno Intervals

- `6.5s` - `59.0s`, rows `106`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.5`, LSTM `0.5884`, XGBoost `0.4457`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `35.0`, recent_utility `1`
- seconds `41.0`, LSTM `0.7771`, XGBoost `0.9179`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `1.5`, LSTM `0.6775`, XGBoost `0.5434`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `39.0`, LSTM `0.5788`, XGBoost `0.4457`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `35.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.6646`, XGBoost `0.5434`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `2.5`, LSTM `0.6624`, XGBoost `0.5434`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `38.0`, LSTM `0.5964`, XGBoost `0.4813`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `35.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.6537`, XGBoost `0.5389`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `39.5`, LSTM `0.5764`, XGBoost `0.4624`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `35.0`, recent_utility `1`
- seconds `36.5`, LSTM `0.5728`, XGBoost `0.4598`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `35.0`, recent_utility `1`
