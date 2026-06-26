# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb/3dmax-vs-vitality-nuke.csv`
- round_num: `7`
- rows: `158`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.524745 | 0.593996 | -0.069250 | 18 | 140 | 0.329114 | 0.841772 |
| active/recent utility | 158 | 1.000 | 0.524745 | 0.593996 | -0.069250 | 18 | 140 | 0.329114 | 0.841772 |
| strong utility action | 139 | 0.880 | 0.534387 | 0.601298 | -0.066911 | 18 | 121 | 0.374101 | 0.820144 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 139 | 0.880 | 0.534387 | 0.601298 | -0.066911 | 18 | 121 | 0.374101 | 0.820144 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 158 | 1.000 | 0.524745 | 0.593996 | -0.069250 | 18 | 140 | 0.329114 | 0.841772 |

## Active Smoke/Inferno Intervals

- `8.0s` - `37.0s`, rows `59`
- `39.0s` - `78.5s`, rows `80`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.0`, LSTM `0.4044`, XGBoost `0.5455`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.4051`, XGBoost `0.5457`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.4033`, XGBoost `0.5434`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4068`, XGBoost `0.5458`, closer `xgboost`, smoke `1`, inferno `5`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4121`, XGBoost `0.5458`, closer `xgboost`, smoke `1`, inferno `5`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.2291`, XGBoost `0.3600`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.4174`, XGBoost `0.5458`, closer `xgboost`, smoke `1`, inferno `5`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.4174`, XGBoost `0.5458`, closer `xgboost`, smoke `1`, inferno `5`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.2352`, XGBoost `0.3600`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.4209`, XGBoost `0.5441`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
