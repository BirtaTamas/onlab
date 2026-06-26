# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `9`
- rows: `134`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 134 | 1.000 | 0.257508 | 0.217126 | 0.040382 | 70 | 64 | 0.619403 | 0.753731 |
| active/recent utility | 134 | 1.000 | 0.257508 | 0.217126 | 0.040382 | 70 | 64 | 0.619403 | 0.753731 |
| strong utility action | 97 | 0.724 | 0.264148 | 0.218796 | 0.045352 | 48 | 49 | 0.628866 | 0.742268 |
| utility damage | 25 | 0.187 | 0.524210 | 0.432819 | 0.091391 | 4 | 21 | 0.320000 | 0.360000 |
| active smoke/inferno | 97 | 0.724 | 0.264148 | 0.218796 | 0.045352 | 48 | 49 | 0.628866 | 0.742268 |
| recent utility last 5s | 10 | 0.075 | 0.521572 | 0.461885 | 0.059686 | 1 | 9 | 0.100000 | 0.700000 |
| flash effect present | 134 | 1.000 | 0.257508 | 0.217126 | 0.040382 | 70 | 64 | 0.619403 | 0.753731 |

## Active Smoke/Inferno Intervals

- `7.5s` - `55.0s`, rows `96`
- `66.5s` - `66.5s`, rows `1`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.5`, LSTM `0.7136`, XGBoost `0.5179`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.7029`, XGBoost `0.5179`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6983`, XGBoost `0.5166`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.6097`, XGBoost `0.4445`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6671`, XGBoost `0.5182`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.6224`, XGBoost `0.4839`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `18.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6867`, XGBoost `0.5518`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `48.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5757`, XGBoost `0.4419`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6541`, XGBoost `0.5212`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6515`, XGBoost `0.5193`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
