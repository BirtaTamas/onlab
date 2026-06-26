# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `32`
- rows: `139`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 139 | 1.000 | 0.695906 | 0.740689 | -0.044783 | 1 | 138 | 0.791367 | 1.000000 |
| active/recent utility | 139 | 1.000 | 0.695906 | 0.740689 | -0.044783 | 1 | 138 | 0.791367 | 1.000000 |
| strong utility action | 123 | 0.885 | 0.721645 | 0.764414 | -0.042769 | 1 | 122 | 0.845528 | 1.000000 |
| utility damage | 10 | 0.072 | 0.500794 | 0.554272 | -0.053478 | 0 | 10 | 0.600000 | 1.000000 |
| active smoke/inferno | 123 | 0.885 | 0.721645 | 0.764414 | -0.042769 | 1 | 122 | 0.845528 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 139 | 1.000 | 0.695906 | 0.740689 | -0.044783 | 1 | 138 | 0.791367 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `69.0s`, rows `123`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.0`, LSTM `0.4591`, XGBoost `0.5541`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4650`, XGBoost `0.5542`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4654`, XGBoost `0.5538`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.4748`, XGBoost `0.5551`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.4748`, XGBoost `0.5536`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.4774`, XGBoost `0.5536`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.4787`, XGBoost `0.5536`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.4796`, XGBoost `0.5535`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.4882`, XGBoost `0.5542`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.6965`, XGBoost `0.7618`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
