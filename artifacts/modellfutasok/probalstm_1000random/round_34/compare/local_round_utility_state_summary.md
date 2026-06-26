# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `6`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.643156 | 0.612158 | 0.030997 | 183 | 47 | 0.978261 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.643156 | 0.612158 | 0.030997 | 183 | 47 | 0.978261 | 1.000000 |
| strong utility action | 185 | 0.804 | 0.637896 | 0.612267 | 0.025629 | 147 | 38 | 0.972973 | 1.000000 |
| utility damage | 10 | 0.043 | 0.657544 | 0.569467 | 0.088077 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 185 | 0.804 | 0.637896 | 0.612267 | 0.025629 | 147 | 38 | 0.972973 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.643156 | 0.612158 | 0.030997 | 183 | 47 | 0.978261 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `59.5s`, rows `103`
- `68.5s` - `93.5s`, rows `51`
- `99.5s` - `114.5s`, rows `31`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `114.5`, LSTM `0.3727`, XGBoost `0.6591`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `114.0`, LSTM `0.3738`, XGBoost `0.6591`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `48.0`, recent_utility `0`
- seconds `113.5`, LSTM `0.3963`, XGBoost `0.6591`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `48.0`, recent_utility `0`
- seconds `113.0`, LSTM `0.4198`, XGBoost `0.6591`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `48.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.6715`, XGBoost `0.9016`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.6682`, XGBoost `0.8959`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `108.5`, LSTM `0.6804`, XGBoost `0.9016`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `112.5`, LSTM `0.4455`, XGBoost `0.6570`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `48.0`, recent_utility `0`
- seconds `111.5`, LSTM `0.6219`, XGBoost `0.8042`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `110.0`, LSTM `0.6840`, XGBoost `0.8638`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `16.0`, recent_utility `0`
