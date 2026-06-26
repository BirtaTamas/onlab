# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-mibr-vs-legacy-nuke-uERfHmzId5aHOSWUmDGvHY/mibr-vs-legacy-nuke.csv`
- round_num: `1`
- rows: `104`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 104 | 1.000 | 0.511821 | 0.542777 | -0.030956 | 21 | 83 | 0.836538 | 0.836538 |
| active/recent utility | 104 | 1.000 | 0.511821 | 0.542777 | -0.030956 | 21 | 83 | 0.836538 | 0.836538 |
| strong utility action | 57 | 0.548 | 0.518082 | 0.562781 | -0.044699 | 6 | 51 | 0.859649 | 0.859649 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 57 | 0.548 | 0.518082 | 0.562781 | -0.044699 | 6 | 51 | 0.859649 | 0.859649 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 104 | 1.000 | 0.511821 | 0.542777 | -0.030956 | 21 | 83 | 0.836538 | 0.836538 |

## Active Smoke/Inferno Intervals

- `12.5s` - `35.5s`, rows `47`
- `47.0s` - `51.5s`, rows `10`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.5`, LSTM `0.5679`, XGBoost `0.8598`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.5706`, XGBoost `0.8554`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5780`, XGBoost `0.8554`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.6043`, XGBoost `0.8598`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5222`, XGBoost `0.7225`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.7667`, XGBoost `0.9352`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5125`, XGBoost `0.6794`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.4401`, XGBoost `0.2920`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.8234`, XGBoost `0.9488`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.4165`, XGBoost `0.2944`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
