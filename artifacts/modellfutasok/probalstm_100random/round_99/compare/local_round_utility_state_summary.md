# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-the-huns-vs-ninja-bo3-8zmdVWrC356tnVH1OFLf2Y/the-huns-vs-ninja-m2-anubis.csv`
- round_num: `9`
- rows: `154`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 154 | 1.000 | 0.011175 | 0.026065 | -0.014889 | 153 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 154 | 1.000 | 0.011175 | 0.026065 | -0.014889 | 153 | 1 | 1.000000 | 1.000000 |
| strong utility action | 110 | 0.714 | 0.011754 | 0.026552 | -0.014798 | 110 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 110 | 0.714 | 0.011754 | 0.026552 | -0.014798 | 110 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 154 | 1.000 | 0.011175 | 0.026065 | -0.014889 | 153 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `62.0s`, rows `110`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.0`, LSTM `0.0092`, XGBoost `0.0667`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.0074`, XGBoost `0.0599`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0069`, XGBoost `0.0592`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.0093`, XGBoost `0.0456`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.0104`, XGBoost `0.0465`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0095`, XGBoost `0.0441`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0100`, XGBoost `0.0442`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0076`, XGBoost `0.0412`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.0113`, XGBoost `0.0439`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0077`, XGBoost `0.0402`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
