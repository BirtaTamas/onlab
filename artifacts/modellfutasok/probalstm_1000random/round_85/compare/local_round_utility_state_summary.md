# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-b8-vs-wildcard-bo3-EO1cCePneo0X8r6rxB_BMC/b8-vs-wildcard-m2-dust2.csv`
- round_num: `1`
- rows: `145`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 145 | 1.000 | 0.375305 | 0.467576 | -0.092271 | 112 | 33 | 0.682759 | 0.503448 |
| active/recent utility | 145 | 1.000 | 0.375305 | 0.467576 | -0.092271 | 112 | 33 | 0.682759 | 0.503448 |
| strong utility action | 90 | 0.621 | 0.425100 | 0.498444 | -0.073343 | 63 | 27 | 0.611111 | 0.344444 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 80 | 0.552 | 0.415070 | 0.497979 | -0.082909 | 62 | 18 | 0.675000 | 0.387500 |
| recent utility last 5s | 10 | 0.069 | 0.505343 | 0.502161 | 0.003182 | 1 | 9 | 0.100000 | 0.000000 |
| flash effect present | 145 | 1.000 | 0.375305 | 0.467576 | -0.092271 | 112 | 33 | 0.682759 | 0.503448 |

## Active Smoke/Inferno Intervals

- `13.0s` - `52.5s`, rows `80`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.5`, LSTM `0.3580`, XGBoost `0.6044`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.6295`, XGBoost `0.8707`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.3574`, XGBoost `0.5897`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.1720`, XGBoost `0.4000`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.6485`, XGBoost `0.8707`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.1694`, XGBoost `0.3747`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.1788`, XGBoost `0.3794`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.2026`, XGBoost `0.3958`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.4667`, XGBoost `0.6527`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.4775`, XGBoost `0.6494`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
