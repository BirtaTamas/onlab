# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `41`
- rows: `246`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 246 | 1.000 | 0.445860 | 0.436780 | 0.009080 | 80 | 166 | 0.280488 | 0.223577 |
| active/recent utility | 246 | 1.000 | 0.445860 | 0.436780 | 0.009080 | 80 | 166 | 0.280488 | 0.223577 |
| strong utility action | 153 | 0.622 | 0.546384 | 0.519634 | 0.026750 | 18 | 135 | 0.045752 | 0.039216 |
| utility damage | 30 | 0.122 | 0.556058 | 0.525091 | 0.030967 | 1 | 29 | 0.000000 | 0.033333 |
| active smoke/inferno | 153 | 0.622 | 0.546384 | 0.519634 | 0.026750 | 18 | 135 | 0.045752 | 0.039216 |
| recent utility last 5s | 10 | 0.041 | 0.580668 | 0.544082 | 0.036586 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 246 | 1.000 | 0.445860 | 0.436780 | 0.009080 | 80 | 166 | 0.280488 | 0.223577 |

## Active Smoke/Inferno Intervals

- `6.0s` - `47.5s`, rows `84`
- `57.5s` - `91.5s`, rows `69`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `75.0`, LSTM `0.5206`, XGBoost `0.3137`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.3303`, XGBoost `0.5216`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.4181`, XGBoost `0.5242`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.2377`, XGBoost `0.1543`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5727`, XGBoost `0.5043`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.0677`, XGBoost `0.1355`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.6061`, XGBoost `0.5428`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5648`, XGBoost `0.5043`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5737`, XGBoost `0.5138`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5709`, XGBoost `0.5132`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
