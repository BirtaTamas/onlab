# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-legacy-vs-vitality-bo3-43WNFDazpfbmBN3Sj5hWmP/vitality-vs-legacy-m2-dust2.csv`
- round_num: `14`
- rows: `142`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.438002 | 0.525542 | -0.087540 | 2 | 140 | 0.239437 | 0.330986 |
| active/recent utility | 142 | 1.000 | 0.438002 | 0.525542 | -0.087540 | 2 | 140 | 0.239437 | 0.330986 |
| strong utility action | 108 | 0.761 | 0.476433 | 0.557541 | -0.081108 | 2 | 106 | 0.305556 | 0.370370 |
| utility damage | 10 | 0.070 | 0.381694 | 0.412879 | -0.031184 | 2 | 8 | 0.000000 | 0.000000 |
| active smoke/inferno | 108 | 0.761 | 0.476433 | 0.557541 | -0.081108 | 2 | 106 | 0.305556 | 0.370370 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 142 | 1.000 | 0.438002 | 0.525542 | -0.087540 | 2 | 140 | 0.239437 | 0.330986 |

## Active Smoke/Inferno Intervals

- `4.0s` - `10.5s`, rows `14`
- `15.0s` - `41.0s`, rows `53`
- `50.5s` - `70.5s`, rows `41`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.0`, LSTM `0.5810`, XGBoost `0.8135`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.4204`, XGBoost `0.6049`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.5542`, XGBoost `0.7278`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.2203`, XGBoost `0.3857`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.2240`, XGBoost `0.3857`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.2252`, XGBoost `0.3866`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.2297`, XGBoost `0.3857`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.2339`, XGBoost `0.3857`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.2647`, XGBoost `0.4125`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.5858`, XGBoost `0.7291`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
