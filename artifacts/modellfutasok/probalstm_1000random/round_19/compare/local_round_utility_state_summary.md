# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-b8-inferno-RJncpU8XKWGlyue1SsisvY/virtus-pro-vs-b8-inferno.csv`
- round_num: `2`
- rows: `208`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 208 | 1.000 | 0.352043 | 0.457728 | -0.105685 | 0 | 208 | 0.216346 | 0.216346 |
| active/recent utility | 208 | 1.000 | 0.352043 | 0.457728 | -0.105685 | 0 | 208 | 0.216346 | 0.216346 |
| strong utility action | 164 | 0.788 | 0.383307 | 0.479146 | -0.095839 | 0 | 164 | 0.250000 | 0.250000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 164 | 0.788 | 0.383307 | 0.479146 | -0.095839 | 0 | 164 | 0.250000 | 0.250000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 208 | 1.000 | 0.352043 | 0.457728 | -0.105685 | 0 | 208 | 0.216346 | 0.216346 |

## Active Smoke/Inferno Intervals

- `10.0s` - `16.5s`, rows `14`
- `20.0s` - `29.0s`, rows `19`
- `32.5s` - `54.0s`, rows `44`
- `58.5s` - `101.5s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `101.0`, LSTM `0.6186`, XGBoost `0.8780`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.6427`, XGBoost `0.8950`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.1291`, XGBoost `0.3244`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.1339`, XGBoost `0.3264`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1343`, XGBoost `0.3264`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.1342`, XGBoost `0.3237`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.1434`, XGBoost `0.3264`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1443`, XGBoost `0.3244`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1444`, XGBoost `0.3244`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.1464`, XGBoost `0.3244`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
