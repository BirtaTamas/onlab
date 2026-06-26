# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `16`
- rows: `257`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 257 | 1.000 | 0.439837 | 0.463606 | -0.023769 | 192 | 65 | 0.237354 | 0.249027 |
| active/recent utility | 257 | 1.000 | 0.439837 | 0.463606 | -0.023769 | 192 | 65 | 0.237354 | 0.249027 |
| strong utility action | 205 | 0.798 | 0.451468 | 0.475379 | -0.023911 | 148 | 57 | 0.224390 | 0.243902 |
| utility damage | 10 | 0.039 | 0.588334 | 0.686246 | -0.097913 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 197 | 0.767 | 0.448707 | 0.471171 | -0.022464 | 140 | 57 | 0.233503 | 0.253807 |
| recent utility last 5s | 10 | 0.039 | 0.519736 | 0.578817 | -0.059081 | 10 | 0 | 0.000000 | 0.000000 |
| flash effect present | 257 | 1.000 | 0.439837 | 0.463606 | -0.023769 | 192 | 65 | 0.237354 | 0.249027 |

## Active Smoke/Inferno Intervals

- `9.5s` - `59.0s`, rows `100`
- `67.5s` - `93.5s`, rows `53`
- `100.0s` - `121.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `103.0`, LSTM `0.4254`, XGBoost `0.1859`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.2064`, XGBoost `0.4251`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `61.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.3995`, XGBoost `0.1908`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.3876`, XGBoost `0.1895`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.4347`, XGBoost `0.2400`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.2367`, XGBoost `0.4247`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `61.0`, recent_utility `0`
- seconds `106.5`, LSTM `0.4241`, XGBoost `0.2400`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.5`, LSTM `0.4106`, XGBoost `0.2283`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.3713`, XGBoost `0.1923`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.3844`, XGBoost `0.2101`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
