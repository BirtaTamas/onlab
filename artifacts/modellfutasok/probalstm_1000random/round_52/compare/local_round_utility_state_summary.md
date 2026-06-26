# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `11`
- rows: `168`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 168 | 1.000 | 0.654998 | 0.708835 | -0.053837 | 29 | 139 | 0.910714 | 0.886905 |
| active/recent utility | 168 | 1.000 | 0.654998 | 0.708835 | -0.053837 | 29 | 139 | 0.910714 | 0.886905 |
| strong utility action | 143 | 0.851 | 0.674157 | 0.734651 | -0.060494 | 15 | 128 | 0.993007 | 0.993007 |
| utility damage | 17 | 0.101 | 0.727068 | 0.698732 | 0.028336 | 12 | 5 | 0.941176 | 0.941176 |
| active smoke/inferno | 142 | 0.845 | 0.674454 | 0.734705 | -0.060250 | 15 | 127 | 0.992958 | 0.992958 |
| recent utility last 5s | 10 | 0.060 | 0.632389 | 0.728135 | -0.095747 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 168 | 1.000 | 0.654998 | 0.708835 | -0.053837 | 29 | 139 | 0.910714 | 0.886905 |

## Active Smoke/Inferno Intervals

- `9.0s` - `48.0s`, rows `79`
- `50.0s` - `78.0s`, rows `57`
- `81.0s` - `83.5s`, rows `6`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.5`, LSTM `0.5628`, XGBoost `0.7387`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5668`, XGBoost `0.7387`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.5734`, XGBoost `0.7387`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5810`, XGBoost `0.7387`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5845`, XGBoost `0.7387`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5770`, XGBoost `0.7282`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5915`, XGBoost `0.7387`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.5763`, XGBoost `0.7233`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.5703`, XGBoost `0.7166`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.5745`, XGBoost `0.7170`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
