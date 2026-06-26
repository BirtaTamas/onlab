# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `10`
- rows: `247`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 247 | 1.000 | 0.245578 | 0.285083 | -0.039505 | 235 | 12 | 1.000000 | 0.939271 |
| active/recent utility | 247 | 1.000 | 0.245578 | 0.285083 | -0.039505 | 235 | 12 | 1.000000 | 0.939271 |
| strong utility action | 197 | 0.798 | 0.293511 | 0.338644 | -0.045133 | 185 | 12 | 1.000000 | 0.954315 |
| utility damage | 10 | 0.040 | 0.471763 | 0.491919 | -0.020156 | 9 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 188 | 0.761 | 0.287579 | 0.330979 | -0.043400 | 176 | 12 | 1.000000 | 0.978723 |
| recent utility last 5s | 10 | 0.040 | 0.414602 | 0.498155 | -0.083553 | 10 | 0 | 1.000000 | 0.500000 |
| flash effect present | 247 | 1.000 | 0.245578 | 0.285083 | -0.039505 | 235 | 12 | 1.000000 | 0.939271 |

## Active Smoke/Inferno Intervals

- `7.5s` - `101.0s`, rows `188`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.5`, LSTM `0.2915`, XGBoost `0.4752`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.3028`, XGBoost `0.4824`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.3018`, XGBoost `0.4752`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.3045`, XGBoost `0.4717`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.3101`, XGBoost `0.4739`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.3191`, XGBoost `0.4737`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.3212`, XGBoost `0.4737`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.3226`, XGBoost `0.4748`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.3287`, XGBoost `0.4757`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.3525`, XGBoost `0.4717`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
