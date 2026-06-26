# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `27`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.558726 | 0.539367 | 0.019359 | 202 | 28 | 0.956522 | 0.704348 |
| active/recent utility | 230 | 1.000 | 0.558726 | 0.539367 | 0.019359 | 202 | 28 | 0.956522 | 0.704348 |
| strong utility action | 177 | 0.770 | 0.562383 | 0.554335 | 0.008048 | 151 | 26 | 0.949153 | 0.836158 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 177 | 0.770 | 0.562383 | 0.554335 | 0.008048 | 151 | 26 | 0.949153 | 0.836158 |
| recent utility last 5s | 10 | 0.043 | 0.581990 | 0.588728 | -0.006738 | 4 | 6 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.558726 | 0.539367 | 0.019359 | 202 | 28 | 0.956522 | 0.704348 |

## Active Smoke/Inferno Intervals

- `6.5s` - `50.5s`, rows `89`
- `51.5s` - `73.5s`, rows `45`
- `93.5s` - `114.5s`, rows `43`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `114.0`, LSTM `0.3416`, XGBoost `0.6545`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `113.5`, LSTM `0.3513`, XGBoost `0.6580`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `113.0`, LSTM `0.3692`, XGBoost `0.6580`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `114.5`, LSTM `0.3761`, XGBoost `0.6574`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `112.5`, LSTM `0.2917`, XGBoost `0.5497`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `112.0`, LSTM `0.3327`, XGBoost `0.5557`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `111.0`, LSTM `0.3424`, XGBoost `0.5503`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.5534`, XGBoost `0.7442`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `111.5`, LSTM `0.3675`, XGBoost `0.5520`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.5484`, XGBoost `0.7147`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
