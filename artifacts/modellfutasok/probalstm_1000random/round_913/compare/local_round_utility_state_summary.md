# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-nrg-bo3-GH6ZBFOA9sfdeCxgnhHN9f/og-vs-nrg-m2-nuke.csv`
- round_num: `11`
- rows: `275`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 275 | 1.000 | 0.194471 | 0.269638 | -0.075167 | 224 | 51 | 0.974545 | 0.960000 |
| active/recent utility | 275 | 1.000 | 0.194471 | 0.269638 | -0.075167 | 224 | 51 | 0.974545 | 0.960000 |
| strong utility action | 242 | 0.880 | 0.188749 | 0.263524 | -0.074775 | 201 | 41 | 0.995868 | 0.971074 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 232 | 0.844 | 0.187188 | 0.259016 | -0.071827 | 191 | 41 | 0.995690 | 0.969828 |
| recent utility last 5s | 10 | 0.036 | 0.224955 | 0.368110 | -0.143155 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 275 | 1.000 | 0.194471 | 0.269638 | -0.075167 | 224 | 51 | 0.974545 | 0.960000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `79.0s`, rows `141`
- `80.0s` - `103.0s`, rows `47`
- `110.5s` - `132.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `119.0`, LSTM `0.0941`, XGBoost `0.3890`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `119.5`, LSTM `0.1019`, XGBoost `0.3890`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `118.5`, LSTM `0.1124`, XGBoost `0.3890`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `120.0`, LSTM `0.1098`, XGBoost `0.3791`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `122.0`, LSTM `0.1118`, XGBoost `0.3771`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `120.5`, LSTM `0.1173`, XGBoost `0.3771`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `121.5`, LSTM `0.1195`, XGBoost `0.3771`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `121.0`, LSTM `0.1209`, XGBoost `0.3771`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `118.0`, LSTM `0.1363`, XGBoost `0.3890`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `124.5`, LSTM `0.1386`, XGBoost `0.3837`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
