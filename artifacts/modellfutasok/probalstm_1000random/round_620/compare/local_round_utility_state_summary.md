# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-g2-bo3-3aFk7fRwd7iUE0VJycUPHK/spirit-vs-g2-m3-ancient.csv`
- round_num: `2`
- rows: `153`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 153 | 1.000 | 0.098766 | 0.121458 | -0.022691 | 136 | 17 | 0.934641 | 0.954248 |
| active/recent utility | 153 | 1.000 | 0.098766 | 0.121458 | -0.022691 | 136 | 17 | 0.934641 | 0.954248 |
| strong utility action | 90 | 0.588 | 0.134607 | 0.159301 | -0.024694 | 74 | 16 | 0.888889 | 0.922222 |
| utility damage | 10 | 0.065 | 0.377471 | 0.369843 | 0.007628 | 5 | 5 | 0.800000 | 1.000000 |
| active smoke/inferno | 90 | 0.588 | 0.134607 | 0.159301 | -0.024694 | 74 | 16 | 0.888889 | 0.922222 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 153 | 1.000 | 0.098766 | 0.121458 | -0.022691 | 136 | 17 | 0.934641 | 0.954248 |

## Active Smoke/Inferno Intervals

- `6.0s` - `28.5s`, rows `46`
- `30.0s` - `51.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.0`, LSTM `0.7409`, XGBoost `0.5888`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.1716`, XGBoost `0.3048`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.0208`, XGBoost `0.1450`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.5488`, XGBoost `0.4252`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.0304`, XGBoost `0.1531`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.0279`, XGBoost `0.1489`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.0241`, XGBoost `0.1451`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.0226`, XGBoost `0.1428`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.0256`, XGBoost `0.1450`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.0264`, XGBoost `0.1450`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
