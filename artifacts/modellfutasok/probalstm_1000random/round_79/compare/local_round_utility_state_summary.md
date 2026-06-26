# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `17`
- rows: `207`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 207 | 1.000 | 0.436473 | 0.521295 | -0.084822 | 4 | 203 | 0.067633 | 0.888889 |
| active/recent utility | 207 | 1.000 | 0.436473 | 0.521295 | -0.084822 | 4 | 203 | 0.067633 | 0.888889 |
| strong utility action | 196 | 0.947 | 0.436101 | 0.521939 | -0.085838 | 4 | 192 | 0.071429 | 0.882653 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 196 | 0.947 | 0.436101 | 0.521939 | -0.085838 | 4 | 192 | 0.071429 | 0.882653 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 207 | 1.000 | 0.436473 | 0.521295 | -0.084822 | 4 | 203 | 0.067633 | 0.888889 |

## Active Smoke/Inferno Intervals

- `3.0s` - `32.0s`, rows `59`
- `35.0s` - `103.0s`, rows `137`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `98.5`, LSTM `0.3092`, XGBoost `0.7196`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.1014`, XGBoost `0.4380`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.1093`, XGBoost `0.4402`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.1265`, XGBoost `0.4405`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.4104`, XGBoost `0.6778`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.1895`, XGBoost `0.4421`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.4775`, XGBoost `0.7119`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.2097`, XGBoost `0.4405`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.4871`, XGBoost `0.7076`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.6319`, XGBoost `0.8510`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
