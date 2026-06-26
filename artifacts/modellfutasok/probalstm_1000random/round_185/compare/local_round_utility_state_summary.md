# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-nrg-bo3-GH6ZBFOA9sfdeCxgnhHN9f/og-vs-nrg-m2-nuke.csv`
- round_num: `6`
- rows: `252`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 252 | 1.000 | 0.470872 | 0.503888 | -0.033015 | 175 | 77 | 0.698413 | 0.285714 |
| active/recent utility | 252 | 1.000 | 0.470872 | 0.503888 | -0.033015 | 175 | 77 | 0.698413 | 0.285714 |
| strong utility action | 134 | 0.532 | 0.500046 | 0.536049 | -0.036003 | 97 | 37 | 0.604478 | 0.089552 |
| utility damage | 10 | 0.040 | 0.474201 | 0.501810 | -0.027609 | 10 | 0 | 1.000000 | 0.000000 |
| active smoke/inferno | 134 | 0.532 | 0.500046 | 0.536049 | -0.036003 | 97 | 37 | 0.604478 | 0.089552 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 252 | 1.000 | 0.470872 | 0.503888 | -0.033015 | 175 | 77 | 0.698413 | 0.285714 |

## Active Smoke/Inferno Intervals

- `9.5s` - `63.0s`, rows `108`
- `76.5s` - `82.0s`, rows `12`
- `96.0s` - `102.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.0`, LSTM `0.2385`, XGBoost `0.5043`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.2413`, XGBoost `0.5043`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.2553`, XGBoost `0.5043`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.2857`, XGBoost `0.5043`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.3404`, XGBoost `0.5104`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.3412`, XGBoost `0.5104`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.3462`, XGBoost `0.5104`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.3465`, XGBoost `0.5104`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.3480`, XGBoost `0.5104`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.3504`, XGBoost `0.5104`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
