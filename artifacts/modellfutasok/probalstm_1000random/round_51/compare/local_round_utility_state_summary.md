# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `3`
- rows: `135`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.943722 | 0.981148 | -0.037425 | 0 | 135 | 1.000000 | 1.000000 |
| active/recent utility | 135 | 1.000 | 0.943722 | 0.981148 | -0.037425 | 0 | 135 | 1.000000 | 1.000000 |
| strong utility action | 113 | 0.837 | 0.940787 | 0.980100 | -0.039313 | 0 | 113 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 103 | 0.763 | 0.941897 | 0.980988 | -0.039091 | 0 | 103 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.074 | 0.929356 | 0.970958 | -0.041601 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 135 | 1.000 | 0.943722 | 0.981148 | -0.037425 | 0 | 135 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `36.5s`, rows `59`
- `40.5s` - `62.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.0`, LSTM `0.9053`, XGBoost `0.9725`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.9059`, XGBoost `0.9725`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.9064`, XGBoost `0.9726`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.9060`, XGBoost `0.9716`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.9088`, XGBoost `0.9723`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.9096`, XGBoost `0.9726`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.9097`, XGBoost `0.9725`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.9099`, XGBoost `0.9724`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.9104`, XGBoost `0.9726`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.9098`, XGBoost `0.9716`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
