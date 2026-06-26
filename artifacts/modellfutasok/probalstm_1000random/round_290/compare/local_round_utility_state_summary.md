# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `15`
- rows: `178`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 178 | 1.000 | 0.526151 | 0.512206 | 0.013945 | 79 | 99 | 0.219101 | 0.247191 |
| active/recent utility | 178 | 1.000 | 0.526151 | 0.512206 | 0.013945 | 79 | 99 | 0.219101 | 0.247191 |
| strong utility action | 163 | 0.916 | 0.521030 | 0.504991 | 0.016040 | 68 | 95 | 0.233129 | 0.263804 |
| utility damage | 10 | 0.056 | 0.489996 | 0.334067 | 0.155929 | 0 | 10 | 0.700000 | 1.000000 |
| active smoke/inferno | 153 | 0.860 | 0.510740 | 0.496417 | 0.014323 | 68 | 85 | 0.248366 | 0.281046 |
| recent utility last 5s | 10 | 0.056 | 0.678475 | 0.636164 | 0.042311 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 178 | 1.000 | 0.526151 | 0.512206 | 0.013945 | 79 | 99 | 0.219101 | 0.247191 |

## Active Smoke/Inferno Intervals

- `7.0s` - `46.0s`, rows `79`
- `51.5s` - `88.0s`, rows `74`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `72.5`, LSTM `0.5038`, XGBoost `0.3247`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.5036`, XGBoost `0.3247`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.5023`, XGBoost `0.3240`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.4983`, XGBoost `0.3287`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.4966`, XGBoost `0.3287`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.4918`, XGBoost `0.3287`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.4851`, XGBoost `0.3287`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.4975`, XGBoost `0.3421`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.4879`, XGBoost `0.3394`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `3.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.4518`, XGBoost `0.3309`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
