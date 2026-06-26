# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-rare-atom-vs-nomads-bo3-2A6RLk5ZJnfAwsBhy_Qbbv/rare-atom-vs-nomads-m1-mirage.csv`
- round_num: `10`
- rows: `118`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 118 | 1.000 | 0.624598 | 0.656810 | -0.032212 | 53 | 65 | 0.194915 | 0.076271 |
| active/recent utility | 118 | 1.000 | 0.624598 | 0.656810 | -0.032212 | 53 | 65 | 0.194915 | 0.076271 |
| strong utility action | 89 | 0.754 | 0.636656 | 0.672629 | -0.035973 | 39 | 50 | 0.213483 | 0.056180 |
| utility damage | 31 | 0.263 | 0.651904 | 0.662518 | -0.010614 | 10 | 21 | 0.096774 | 0.096774 |
| active smoke/inferno | 89 | 0.754 | 0.636656 | 0.672629 | -0.035973 | 39 | 50 | 0.213483 | 0.056180 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 118 | 1.000 | 0.624598 | 0.656810 | -0.032212 | 53 | 65 | 0.194915 | 0.076271 |

## Active Smoke/Inferno Intervals

- `7.0s` - `51.0s`, rows `89`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.0`, LSTM `0.3750`, XGBoost `0.6189`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.4076`, XGBoost `0.6228`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.4093`, XGBoost `0.6240`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4266`, XGBoost `0.6349`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5629`, XGBoost `0.7675`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5796`, XGBoost `0.7790`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5804`, XGBoost `0.7790`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5562`, XGBoost `0.7536`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.3527`, XGBoost `0.5498`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.5754`, XGBoost `0.7675`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
