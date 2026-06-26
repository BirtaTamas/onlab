# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb/3dmax-vs-vitality-nuke.csv`
- round_num: `6`
- rows: `185`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 185 | 1.000 | 0.259903 | 0.261979 | -0.002076 | 133 | 52 | 0.654054 | 0.654054 |
| active/recent utility | 185 | 1.000 | 0.259903 | 0.261979 | -0.002076 | 133 | 52 | 0.654054 | 0.654054 |
| strong utility action | 147 | 0.795 | 0.257337 | 0.260660 | -0.003323 | 107 | 40 | 0.673469 | 0.673469 |
| utility damage | 20 | 0.108 | 0.265120 | 0.288784 | -0.023664 | 20 | 0 | 0.500000 | 0.500000 |
| active smoke/inferno | 137 | 0.741 | 0.276034 | 0.279354 | -0.003320 | 97 | 40 | 0.649635 | 0.649635 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 185 | 1.000 | 0.259903 | 0.261979 | -0.002076 | 133 | 52 | 0.654054 | 0.654054 |

## Active Smoke/Inferno Intervals

- `8.0s` - `37.0s`, rows `59`
- `39.0s` - `77.5s`, rows `78`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.5`, LSTM `0.3929`, XGBoost `0.2928`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.1750`, XGBoost `0.0793`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.4011`, XGBoost `0.3058`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.3911`, XGBoost `0.2973`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.3868`, XGBoost `0.2935`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.3807`, XGBoost `0.2959`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.3770`, XGBoost `0.2935`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.3793`, XGBoost `0.2973`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.3741`, XGBoost `0.2962`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.3679`, XGBoost `0.2935`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
