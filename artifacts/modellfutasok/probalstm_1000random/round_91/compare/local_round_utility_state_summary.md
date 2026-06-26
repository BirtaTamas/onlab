# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-nrg-dust2-QDtqFlW1Z9UhZpBNOAavnd/heroic-vs-nrg-dust2.csv`
- round_num: `1`
- rows: `138`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 138 | 1.000 | 0.259808 | 0.335034 | -0.075226 | 125 | 13 | 0.840580 | 0.768116 |
| active/recent utility | 138 | 1.000 | 0.259808 | 0.335034 | -0.075226 | 125 | 13 | 0.840580 | 0.768116 |
| strong utility action | 51 | 0.370 | 0.245090 | 0.312517 | -0.067427 | 47 | 4 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 51 | 0.370 | 0.245090 | 0.312517 | -0.067427 | 47 | 4 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 138 | 1.000 | 0.259808 | 0.335034 | -0.075226 | 125 | 13 | 0.840580 | 0.768116 |

## Active Smoke/Inferno Intervals

- `32.0s` - `57.0s`, rows `51`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.5`, LSTM `0.2400`, XGBoost `0.4858`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.2490`, XGBoost `0.4847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.2584`, XGBoost `0.4849`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.2679`, XGBoost `0.4849`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.2914`, XGBoost `0.4849`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.2952`, XGBoost `0.4847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.3217`, XGBoost `0.4847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.3393`, XGBoost `0.4858`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.3799`, XGBoost `0.4847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.3293`, XGBoost `0.4327`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
