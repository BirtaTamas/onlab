# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-3dmax-vs-lynn-vision-bo3-0ZNMTlQ0ZfadRgwA0Ax5fN/3dmax-vs-lynn-vision-m2-anubis.csv`
- round_num: `3`
- rows: `240`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 240 | 1.000 | 0.013547 | 0.046083 | -0.032536 | 239 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 240 | 1.000 | 0.013547 | 0.046083 | -0.032536 | 239 | 1 | 1.000000 | 1.000000 |
| strong utility action | 179 | 0.746 | 0.013350 | 0.047386 | -0.034036 | 179 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 179 | 0.746 | 0.013350 | 0.047386 | -0.034036 | 179 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.042 | 0.020753 | 0.060013 | -0.039260 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 240 | 1.000 | 0.013547 | 0.046083 | -0.032536 | 239 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `56.5s`, rows `95`
- `65.0s` - `106.5s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `56.5`, LSTM `0.0153`, XGBoost `0.0729`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0096`, XGBoost `0.0667`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.0106`, XGBoost `0.0677`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.0121`, XGBoost `0.0677`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.0199`, XGBoost `0.0733`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.0149`, XGBoost `0.0679`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.0214`, XGBoost `0.0732`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.0168`, XGBoost `0.0677`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.0173`, XGBoost `0.0677`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.0232`, XGBoost `0.0731`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
