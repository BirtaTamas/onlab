# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `3`
- rows: `139`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 139 | 1.000 | 0.157166 | 0.230603 | -0.073437 | 139 | 0 | 1.000000 | 0.942446 |
| active/recent utility | 139 | 1.000 | 0.157166 | 0.230603 | -0.073437 | 139 | 0 | 1.000000 | 0.942446 |
| strong utility action | 108 | 0.777 | 0.148315 | 0.222550 | -0.074235 | 108 | 0 | 1.000000 | 0.962963 |
| utility damage | 21 | 0.151 | 0.153013 | 0.237669 | -0.084657 | 21 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 108 | 0.777 | 0.148315 | 0.222550 | -0.074235 | 108 | 0 | 1.000000 | 0.962963 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 139 | 1.000 | 0.157166 | 0.230603 | -0.073437 | 139 | 0 | 1.000000 | 0.942446 |

## Active Smoke/Inferno Intervals

- `8.0s` - `61.5s`, rows `108`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `16.5`, LSTM `0.1190`, XGBoost `0.3537`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.1278`, XGBoost `0.3525`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.1261`, XGBoost `0.3453`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.1280`, XGBoost `0.3458`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.1412`, XGBoost `0.3574`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.2113`, XGBoost `0.4165`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1668`, XGBoost `0.3600`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `14.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.1889`, XGBoost `0.3814`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.1664`, XGBoost `0.3582`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.1937`, XGBoost `0.3835`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
