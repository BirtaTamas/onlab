# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-metizport-inferno-qyaWW06KtkktSDfICHvaab/wildcard-vs-metizport-inferno.csv`
- round_num: `7`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.735731 | 0.698760 | 0.036971 | 156 | 74 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.735731 | 0.698760 | 0.036971 | 156 | 74 | 1.000000 | 1.000000 |
| strong utility action | 178 | 0.774 | 0.713302 | 0.672284 | 0.041018 | 130 | 48 | 1.000000 | 1.000000 |
| utility damage | 13 | 0.057 | 0.637504 | 0.554726 | 0.082778 | 13 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 178 | 0.774 | 0.713302 | 0.672284 | 0.041018 | 130 | 48 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.735731 | 0.698760 | 0.036971 | 156 | 74 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `75.5s`, rows `131`
- `82.5s` - `105.5s`, rows `47`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.6351`, XGBoost `0.5237`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6312`, XGBoost `0.5235`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6306`, XGBoost `0.5235`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.6598`, XGBoost `0.5529`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.6593`, XGBoost `0.5529`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.6579`, XGBoost `0.5529`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.6281`, XGBoost `0.5235`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.6558`, XGBoost `0.5529`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.6277`, XGBoost `0.5249`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6280`, XGBoost `0.5275`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
