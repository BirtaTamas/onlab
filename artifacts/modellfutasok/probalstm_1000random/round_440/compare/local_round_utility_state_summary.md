# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-the-huns-vs-ninja-bo3-8zmdVWrC356tnVH1OFLf2Y/the-huns-vs-ninja-m2-anubis.csv`
- round_num: `8`
- rows: `131`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 131 | 1.000 | 0.335154 | 0.336406 | -0.001253 | 64 | 67 | 0.511450 | 0.541985 |
| active/recent utility | 131 | 1.000 | 0.335154 | 0.336406 | -0.001253 | 64 | 67 | 0.511450 | 0.541985 |
| strong utility action | 103 | 0.786 | 0.276412 | 0.277681 | -0.001268 | 54 | 49 | 0.640777 | 0.689320 |
| utility damage | 32 | 0.244 | 0.386071 | 0.353482 | 0.032588 | 12 | 20 | 0.437500 | 0.593750 |
| active smoke/inferno | 103 | 0.786 | 0.276412 | 0.277681 | -0.001268 | 54 | 49 | 0.640777 | 0.689320 |
| recent utility last 5s | 10 | 0.076 | 0.068562 | 0.075300 | -0.006738 | 6 | 4 | 1.000000 | 1.000000 |
| flash effect present | 131 | 1.000 | 0.335154 | 0.336406 | -0.001253 | 64 | 67 | 0.511450 | 0.541985 |

## Active Smoke/Inferno Intervals

- `14.0s` - `65.0s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `61.5`, LSTM `0.5448`, XGBoost `0.3545`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.5316`, XGBoost `0.3545`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.5297`, XGBoost `0.3545`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.5243`, XGBoost `0.3545`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.5235`, XGBoost `0.3545`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5042`, XGBoost `0.3545`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.4793`, XGBoost `0.3510`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.4659`, XGBoost `0.3510`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `9.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.7223`, XGBoost `0.6229`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.2814`, XGBoost `0.1865`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `40.0`, recent_utility `0`
