# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `9`
- rows: `138`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 138 | 1.000 | 0.739877 | 0.733165 | 0.006712 | 89 | 49 | 1.000000 | 1.000000 |
| active/recent utility | 138 | 1.000 | 0.739877 | 0.733165 | 0.006712 | 89 | 49 | 1.000000 | 1.000000 |
| strong utility action | 118 | 0.855 | 0.750364 | 0.743462 | 0.006902 | 76 | 42 | 1.000000 | 1.000000 |
| utility damage | 28 | 0.203 | 0.775230 | 0.788050 | -0.012820 | 13 | 15 | 1.000000 | 1.000000 |
| active smoke/inferno | 118 | 0.855 | 0.750364 | 0.743462 | 0.006902 | 76 | 42 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 138 | 1.000 | 0.739877 | 0.733165 | 0.006712 | 89 | 49 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `36.5s`, rows `58`
- `39.0s` - `68.5s`, rows `60`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.0`, LSTM `0.7319`, XGBoost `0.6590`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.8589`, XGBoost `0.9310`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.7063`, XGBoost `0.6361`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.8621`, XGBoost `0.9304`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `50.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.8631`, XGBoost `0.9310`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `44.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.7008`, XGBoost `0.6353`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.8667`, XGBoost `0.9310`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.6984`, XGBoost `0.6354`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.6976`, XGBoost `0.6388`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.6908`, XGBoost `0.6354`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
