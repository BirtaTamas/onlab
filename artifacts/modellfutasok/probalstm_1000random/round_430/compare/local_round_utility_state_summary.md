# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `1`
- rows: `150`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 150 | 1.000 | 0.477430 | 0.567834 | -0.090404 | 9 | 141 | 0.373333 | 0.846667 |
| active/recent utility | 150 | 1.000 | 0.477430 | 0.567834 | -0.090404 | 9 | 141 | 0.373333 | 0.846667 |
| strong utility action | 58 | 0.387 | 0.446642 | 0.559070 | -0.112428 | 0 | 58 | 0.310345 | 0.810345 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 58 | 0.387 | 0.446642 | 0.559070 | -0.112428 | 0 | 58 | 0.310345 | 0.810345 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 150 | 1.000 | 0.477430 | 0.567834 | -0.090404 | 9 | 141 | 0.373333 | 0.846667 |

## Active Smoke/Inferno Intervals

- `28.0s` - `56.5s`, rows `58`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `56.0`, LSTM `0.5010`, XGBoost `0.7214`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.4912`, XGBoost `0.7059`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.4932`, XGBoost `0.7026`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.4860`, XGBoost `0.6902`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.5736`, XGBoost `0.7712`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5588`, XGBoost `0.7535`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5521`, XGBoost `0.7408`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5305`, XGBoost `0.7163`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.5454`, XGBoost `0.7239`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5926`, XGBoost `0.7709`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
