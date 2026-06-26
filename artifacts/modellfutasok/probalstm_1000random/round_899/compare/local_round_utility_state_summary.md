# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `34`
- rows: `169`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 169 | 1.000 | 0.210693 | 0.215862 | -0.005170 | 125 | 44 | 0.644970 | 0.633136 |
| active/recent utility | 169 | 1.000 | 0.210693 | 0.215862 | -0.005170 | 125 | 44 | 0.644970 | 0.633136 |
| strong utility action | 120 | 0.710 | 0.278747 | 0.284047 | -0.005300 | 79 | 41 | 0.525000 | 0.516667 |
| utility damage | 20 | 0.118 | 0.210499 | 0.219335 | -0.008836 | 12 | 8 | 0.700000 | 0.750000 |
| active smoke/inferno | 110 | 0.651 | 0.255633 | 0.261965 | -0.006332 | 78 | 32 | 0.572727 | 0.563636 |
| recent utility last 5s | 20 | 0.118 | 0.282991 | 0.301912 | -0.018921 | 9 | 11 | 0.500000 | 0.500000 |
| flash effect present | 169 | 1.000 | 0.210693 | 0.215862 | -0.005170 | 125 | 44 | 0.644970 | 0.633136 |

## Active Smoke/Inferno Intervals

- `7.0s` - `61.5s`, rows `110`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.0`, LSTM `0.0807`, XGBoost `0.2530`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `33.5`, LSTM `0.1048`, XGBoost `0.2459`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `32.5`, LSTM `0.1234`, XGBoost `0.2298`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.1329`, XGBoost `0.2364`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.6429`, XGBoost `0.7225`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5426`, XGBoost `0.4809`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.1702`, XGBoost `0.2311`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5652`, XGBoost `0.5090`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5788`, XGBoost `0.5228`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.5779`, XGBoost `0.5228`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
