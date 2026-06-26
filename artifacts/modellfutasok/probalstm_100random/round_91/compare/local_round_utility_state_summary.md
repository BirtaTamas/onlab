# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv`
- round_num: `12`
- rows: `145`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 145 | 1.000 | 0.487027 | 0.594333 | -0.107306 | 140 | 5 | 0.510345 | 0.344828 |
| active/recent utility | 145 | 1.000 | 0.487027 | 0.594333 | -0.107306 | 140 | 5 | 0.510345 | 0.344828 |
| strong utility action | 98 | 0.676 | 0.484587 | 0.591436 | -0.106848 | 95 | 3 | 0.540816 | 0.336735 |
| utility damage | 10 | 0.069 | 0.551159 | 0.696797 | -0.145638 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 98 | 0.676 | 0.484587 | 0.591436 | -0.106848 | 95 | 3 | 0.540816 | 0.336735 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 145 | 1.000 | 0.487027 | 0.594333 | -0.107306 | 140 | 5 | 0.510345 | 0.344828 |

## Active Smoke/Inferno Intervals

- `7.5s` - `45.0s`, rows `76`
- `58.5s` - `67.5s`, rows `19`
- `71.0s` - `72.0s`, rows `3`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.0`, LSTM `0.6252`, XGBoost `0.9071`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.5409`, XGBoost `0.8211`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.5503`, XGBoost `0.8140`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.3870`, XGBoost `0.6110`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.5274`, XGBoost `0.7480`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.4015`, XGBoost `0.6115`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5177`, XGBoost `0.7092`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.3561`, XGBoost `0.5465`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.4252`, XGBoost `0.6126`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5221`, XGBoost `0.7092`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
