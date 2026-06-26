# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `24`
- rows: `235`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 235 | 1.000 | 0.338639 | 0.359518 | -0.020878 | 166 | 69 | 0.791489 | 0.770213 |
| active/recent utility | 235 | 1.000 | 0.338639 | 0.359518 | -0.020878 | 166 | 69 | 0.791489 | 0.770213 |
| strong utility action | 162 | 0.689 | 0.451155 | 0.465614 | -0.014459 | 93 | 69 | 0.697531 | 0.759259 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 162 | 0.689 | 0.451155 | 0.465614 | -0.014459 | 93 | 69 | 0.697531 | 0.759259 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 235 | 1.000 | 0.338639 | 0.359518 | -0.020878 | 166 | 69 | 0.791489 | 0.770213 |

## Active Smoke/Inferno Intervals

- `7.5s` - `88.0s`, rows `162`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.5`, LSTM `0.3607`, XGBoost `0.5103`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3654`, XGBoost `0.5109`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.3670`, XGBoost `0.5106`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.3663`, XGBoost `0.5088`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.3692`, XGBoost `0.5106`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.3714`, XGBoost `0.5106`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.3749`, XGBoost `0.5070`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.3803`, XGBoost `0.5110`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.3883`, XGBoost `0.5182`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.3911`, XGBoost `0.5148`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
