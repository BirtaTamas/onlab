# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `2`
- rows: `173`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.728172 | 0.793493 | -0.065321 | 163 | 10 | 0.156069 | 0.156069 |
| active/recent utility | 173 | 1.000 | 0.728172 | 0.793493 | -0.065321 | 163 | 10 | 0.156069 | 0.156069 |
| strong utility action | 109 | 0.630 | 0.770351 | 0.843798 | -0.073447 | 105 | 4 | 0.091743 | 0.091743 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 99 | 0.572 | 0.844589 | 0.920847 | -0.076258 | 95 | 4 | 0.000000 | 0.000000 |
| recent utility last 5s | 10 | 0.058 | 0.035394 | 0.081009 | -0.045615 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 173 | 1.000 | 0.728172 | 0.793493 | -0.065321 | 163 | 10 | 0.156069 | 0.156069 |

## Active Smoke/Inferno Intervals

- `13.0s` - `34.5s`, rows `44`
- `40.0s` - `61.5s`, rows `44`
- `62.5s` - `67.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.5`, LSTM `0.7895`, XGBoost `0.9689`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.7926`, XGBoost `0.9692`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.7946`, XGBoost `0.9691`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.7963`, XGBoost `0.9691`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.8005`, XGBoost `0.9689`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.8008`, XGBoost `0.9690`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.8025`, XGBoost `0.9689`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.8046`, XGBoost `0.9689`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.8066`, XGBoost `0.9688`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.8083`, XGBoost `0.9689`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
