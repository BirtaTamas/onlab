# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-faze-bo3-ZgdBOa3Yi0KCkwa_Ap1ef3/aurora-vs-faze-m2-train.csv`
- round_num: `15`
- rows: `190`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 190 | 1.000 | 0.711838 | 0.730785 | -0.018946 | 61 | 129 | 0.952632 | 0.952632 |
| active/recent utility | 190 | 1.000 | 0.711838 | 0.730785 | -0.018946 | 61 | 129 | 0.952632 | 0.952632 |
| strong utility action | 167 | 0.879 | 0.710998 | 0.737302 | -0.026304 | 46 | 121 | 0.946108 | 0.946108 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 167 | 0.879 | 0.710998 | 0.737302 | -0.026304 | 46 | 121 | 0.946108 | 0.946108 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 190 | 1.000 | 0.711838 | 0.730785 | -0.018946 | 61 | 129 | 0.952632 | 0.952632 |

## Active Smoke/Inferno Intervals

- `7.5s` - `86.5s`, rows `159`
- `91.0s` - `94.5s`, rows `8`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.5`, LSTM `0.4243`, XGBoost `0.2208`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.4162`, XGBoost `0.2262`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.4072`, XGBoost `0.2262`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.4049`, XGBoost `0.2277`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.7137`, XGBoost `0.8602`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.7178`, XGBoost `0.8628`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.7173`, XGBoost `0.8602`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.7173`, XGBoost `0.8602`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.7162`, XGBoost `0.8587`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.7205`, XGBoost `0.8628`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
