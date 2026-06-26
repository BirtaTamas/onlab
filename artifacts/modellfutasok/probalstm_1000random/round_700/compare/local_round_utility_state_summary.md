# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv`
- round_num: `7`
- rows: `208`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 208 | 1.000 | 0.525053 | 0.687549 | -0.162497 | 208 | 0 | 0.370192 | 0.000000 |
| active/recent utility | 208 | 1.000 | 0.525053 | 0.687549 | -0.162497 | 208 | 0 | 0.370192 | 0.000000 |
| strong utility action | 192 | 0.923 | 0.528791 | 0.688463 | -0.159671 | 192 | 0 | 0.338542 | 0.000000 |
| utility damage | 10 | 0.048 | 0.783810 | 0.875165 | -0.091355 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 192 | 0.923 | 0.528791 | 0.688463 | -0.159671 | 192 | 0 | 0.338542 | 0.000000 |
| recent utility last 5s | 17 | 0.082 | 0.355301 | 0.516166 | -0.160866 | 17 | 0 | 1.000000 | 0.000000 |
| flash effect present | 208 | 1.000 | 0.525053 | 0.687549 | -0.162497 | 208 | 0 | 0.370192 | 0.000000 |

## Active Smoke/Inferno Intervals

- `2.5s` - `98.0s`, rows `192`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.5`, LSTM `0.4983`, XGBoost `0.7186`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.4982`, XGBoost `0.7173`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.4997`, XGBoost `0.7185`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.4995`, XGBoost `0.7182`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.4987`, XGBoost `0.7173`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.4994`, XGBoost `0.7173`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.5006`, XGBoost `0.7182`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.5018`, XGBoost `0.7182`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.5023`, XGBoost `0.7182`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.5027`, XGBoost `0.7186`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
