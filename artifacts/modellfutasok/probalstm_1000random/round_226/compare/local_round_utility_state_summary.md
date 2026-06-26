# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `1`
- rows: `170`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 170 | 1.000 | 0.347633 | 0.409940 | -0.062307 | 152 | 18 | 0.717647 | 0.323529 |
| active/recent utility | 170 | 1.000 | 0.347633 | 0.409940 | -0.062307 | 152 | 18 | 0.717647 | 0.323529 |
| strong utility action | 89 | 0.524 | 0.336536 | 0.420101 | -0.083565 | 86 | 3 | 0.764045 | 0.393258 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 89 | 0.524 | 0.336536 | 0.420101 | -0.083565 | 86 | 3 | 0.764045 | 0.393258 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 170 | 1.000 | 0.347633 | 0.409940 | -0.062307 | 152 | 18 | 0.717647 | 0.323529 |

## Active Smoke/Inferno Intervals

- `7.5s` - `29.5s`, rows `45`
- `53.0s` - `74.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.0`, LSTM `0.1691`, XGBoost `0.5216`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.1888`, XGBoost `0.5216`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.1358`, XGBoost `0.4685`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.1352`, XGBoost `0.4667`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.1990`, XGBoost `0.5216`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.1468`, XGBoost `0.4682`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.2456`, XGBoost `0.5216`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.2502`, XGBoost `0.5216`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.2537`, XGBoost `0.5216`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.2619`, XGBoost `0.5216`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
