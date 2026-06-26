# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `11`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.866330 | 0.855084 | 0.011246 | 148 | 82 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.866330 | 0.855084 | 0.011246 | 148 | 82 | 1.000000 | 1.000000 |
| strong utility action | 103 | 0.448 | 0.816002 | 0.824631 | -0.008630 | 29 | 74 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.087 | 0.705091 | 0.673249 | 0.031843 | 16 | 4 | 1.000000 | 1.000000 |
| active smoke/inferno | 103 | 0.448 | 0.816002 | 0.824631 | -0.008630 | 29 | 74 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.866330 | 0.855084 | 0.011246 | 148 | 82 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `61.5s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.6521`, XGBoost `0.5313`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6344`, XGBoost `0.5315`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.7047`, XGBoost `0.6244`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `116.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.7018`, XGBoost `0.6271`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `116.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6970`, XGBoost `0.6244`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `116.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.7071`, XGBoost `0.6367`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `116.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.6972`, XGBoost `0.6271`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `116.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6928`, XGBoost `0.6271`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `116.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.9021`, XGBoost `0.9575`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.9021`, XGBoost `0.9561`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
