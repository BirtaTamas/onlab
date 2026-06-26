# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `13`
- rows: `215`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 215 | 1.000 | 0.815917 | 0.880174 | -0.064256 | 0 | 215 | 1.000000 | 1.000000 |
| active/recent utility | 215 | 1.000 | 0.815917 | 0.880174 | -0.064256 | 0 | 215 | 1.000000 | 1.000000 |
| strong utility action | 111 | 0.516 | 0.890790 | 0.958205 | -0.067415 | 0 | 111 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 111 | 0.516 | 0.890790 | 0.958205 | -0.067415 | 0 | 111 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 215 | 1.000 | 0.815917 | 0.880174 | -0.064256 | 0 | 215 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `22.5s` - `44.0s`, rows `44`
- `59.0s` - `85.0s`, rows `53`
- `90.5s` - `97.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `96.5`, LSTM `0.8196`, XGBoost `0.9484`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.8512`, XGBoost `0.9566`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.8542`, XGBoost `0.9561`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.8551`, XGBoost `0.9566`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.8571`, XGBoost `0.9559`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.8600`, XGBoost `0.9575`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.8624`, XGBoost `0.9563`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.8623`, XGBoost `0.9561`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.8647`, XGBoost `0.9579`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.8637`, XGBoost `0.9563`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
