# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `6`
- rows: `278`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 278 | 1.000 | 0.288770 | 0.191755 | 0.097015 | 149 | 129 | 0.579137 | 1.000000 |
| active/recent utility | 278 | 1.000 | 0.288770 | 0.191755 | 0.097015 | 149 | 129 | 0.579137 | 1.000000 |
| strong utility action | 117 | 0.421 | 0.319556 | 0.225370 | 0.094185 | 53 | 64 | 0.547009 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 117 | 0.421 | 0.319556 | 0.225370 | 0.094185 | 53 | 64 | 0.547009 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 278 | 1.000 | 0.288770 | 0.191755 | 0.097015 | 149 | 129 | 0.579137 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `51.0s`, rows `89`
- `89.5s` - `96.0s`, rows `14`
- `101.5s` - `108.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.5`, LSTM `0.6383`, XGBoost `0.2939`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.6161`, XGBoost `0.2939`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.6165`, XGBoost `0.2981`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.6137`, XGBoost `0.2981`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.6130`, XGBoost `0.2981`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.6121`, XGBoost `0.2985`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.6116`, XGBoost `0.2981`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.6077`, XGBoost `0.2981`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.6104`, XGBoost `0.3016`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.6060`, XGBoost `0.2990`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
