# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv`
- round_num: `19`
- rows: `212`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 212 | 1.000 | 0.591574 | 0.637296 | -0.045722 | 4 | 208 | 1.000000 | 1.000000 |
| active/recent utility | 212 | 1.000 | 0.591574 | 0.637296 | -0.045722 | 4 | 208 | 1.000000 | 1.000000 |
| strong utility action | 137 | 0.646 | 0.605962 | 0.657323 | -0.051361 | 1 | 136 | 1.000000 | 1.000000 |
| utility damage | 12 | 0.057 | 0.539506 | 0.565790 | -0.026285 | 0 | 12 | 1.000000 | 1.000000 |
| active smoke/inferno | 137 | 0.646 | 0.605962 | 0.657323 | -0.051361 | 1 | 136 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 212 | 1.000 | 0.591574 | 0.637296 | -0.045722 | 4 | 208 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `54.0s`, rows `93`
- `84.0s` - `105.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `104.0`, LSTM `0.8011`, XGBoost `0.9308`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.6133`, XGBoost `0.7385`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.8120`, XGBoost `0.9338`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.8122`, XGBoost `0.9338`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.8173`, XGBoost `0.9357`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.8165`, XGBoost `0.9337`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.6215`, XGBoost `0.7385`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.8223`, XGBoost `0.9308`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.6302`, XGBoost `0.7381`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.6324`, XGBoost `0.7385`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `10.0`, recent_utility `0`
