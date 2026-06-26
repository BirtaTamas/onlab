# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `11`
- rows: `235`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 235 | 1.000 | 0.461196 | 0.449554 | 0.011642 | 120 | 115 | 0.438298 | 0.387234 |
| active/recent utility | 235 | 1.000 | 0.461196 | 0.449554 | 0.011642 | 120 | 115 | 0.438298 | 0.387234 |
| strong utility action | 151 | 0.643 | 0.620748 | 0.603580 | 0.017169 | 55 | 96 | 0.251656 | 0.172185 |
| utility damage | 10 | 0.043 | 0.758606 | 0.737787 | 0.020819 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 151 | 0.643 | 0.620748 | 0.603580 | 0.017169 | 55 | 96 | 0.251656 | 0.172185 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 235 | 1.000 | 0.461196 | 0.449554 | 0.011642 | 120 | 115 | 0.438298 | 0.387234 |

## Active Smoke/Inferno Intervals

- `6.5s` - `34.5s`, rows `57`
- `38.0s` - `84.5s`, rows `94`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `16.0`, LSTM `0.8555`, XGBoost `0.7187`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.4163`, XGBoost `0.5456`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.8417`, XGBoost `0.7171`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.8376`, XGBoost `0.7169`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.7971`, XGBoost `0.6780`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.8249`, XGBoost `0.7148`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.8232`, XGBoost `0.7143`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.8214`, XGBoost `0.7143`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.4381`, XGBoost `0.5449`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.8212`, XGBoost `0.7147`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
