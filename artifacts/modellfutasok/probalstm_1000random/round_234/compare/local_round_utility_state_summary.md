# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `5`
- rows: `264`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 264 | 1.000 | 0.373490 | 0.396842 | -0.023352 | 204 | 60 | 0.787879 | 0.666667 |
| active/recent utility | 264 | 1.000 | 0.373490 | 0.396842 | -0.023352 | 204 | 60 | 0.787879 | 0.666667 |
| strong utility action | 198 | 0.750 | 0.382963 | 0.410687 | -0.027724 | 160 | 38 | 0.787879 | 0.656566 |
| utility damage | 40 | 0.152 | 0.500883 | 0.542724 | -0.041841 | 37 | 3 | 0.425000 | 0.200000 |
| active smoke/inferno | 192 | 0.727 | 0.377629 | 0.405664 | -0.028035 | 155 | 37 | 0.812500 | 0.677083 |
| recent utility last 5s | 10 | 0.038 | 0.415985 | 0.490148 | -0.074162 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 264 | 1.000 | 0.373490 | 0.396842 | -0.023352 | 204 | 60 | 0.787879 | 0.666667 |

## Active Smoke/Inferno Intervals

- `9.0s` - `53.0s`, rows `89`
- `66.5s` - `117.5s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `91.0`, LSTM `0.4487`, XGBoost `0.2563`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.4332`, XGBoost `0.2563`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.4325`, XGBoost `0.2563`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.4287`, XGBoost `0.2567`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.4096`, XGBoost `0.2567`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.4012`, XGBoost `0.2567`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.4006`, XGBoost `0.2567`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.3703`, XGBoost `0.2415`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.4461`, XGBoost `0.5661`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.3594`, XGBoost `0.2415`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
