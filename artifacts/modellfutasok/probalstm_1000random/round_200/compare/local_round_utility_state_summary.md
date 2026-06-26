# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-legacy-bo3-NvI4DRplwm0O-zy6YVkFbj/wildcard-vs-legacy-m2-nuke.csv`
- round_num: `13`
- rows: `170`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 170 | 1.000 | 0.179265 | 0.191497 | -0.012232 | 137 | 33 | 0.688235 | 0.847059 |
| active/recent utility | 170 | 1.000 | 0.179265 | 0.191497 | -0.012232 | 137 | 33 | 0.688235 | 0.847059 |
| strong utility action | 44 | 0.259 | 0.149522 | 0.186695 | -0.037173 | 33 | 11 | 0.795455 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.259 | 0.149522 | 0.186695 | -0.037173 | 33 | 11 | 0.795455 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 170 | 1.000 | 0.179265 | 0.191497 | -0.012232 | 137 | 33 | 0.688235 | 0.847059 |

## Active Smoke/Inferno Intervals

- `23.0s` - `44.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.0`, LSTM `0.1046`, XGBoost `0.2276`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.1060`, XGBoost `0.2276`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.3451`, XGBoost `0.2297`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.1193`, XGBoost `0.2276`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.1223`, XGBoost `0.2276`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0097`, XGBoost `0.0789`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0297`, XGBoost `0.0988`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0098`, XGBoost `0.0789`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.0128`, XGBoost `0.0809`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.0121`, XGBoost `0.0799`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
