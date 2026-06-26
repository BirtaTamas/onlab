# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `15`
- rows: `204`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 204 | 1.000 | 0.656133 | 0.623527 | 0.032606 | 146 | 58 | 0.970588 | 0.970588 |
| active/recent utility | 204 | 1.000 | 0.656133 | 0.623527 | 0.032606 | 146 | 58 | 0.970588 | 0.970588 |
| strong utility action | 169 | 0.828 | 0.622218 | 0.583500 | 0.038717 | 136 | 33 | 0.964497 | 0.964497 |
| utility damage | 14 | 0.069 | 0.592168 | 0.546291 | 0.045877 | 14 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 157 | 0.770 | 0.621140 | 0.587279 | 0.033861 | 124 | 33 | 0.961783 | 0.961783 |
| recent utility last 5s | 12 | 0.059 | 0.636317 | 0.534067 | 0.102249 | 12 | 0 | 1.000000 | 1.000000 |
| flash effect present | 204 | 1.000 | 0.656133 | 0.623527 | 0.032606 | 146 | 58 | 0.970588 | 0.970588 |

## Active Smoke/Inferno Intervals

- `8.5s` - `81.0s`, rows `146`
- `94.0s` - `99.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `70.5`, LSTM `0.4576`, XGBoost `0.3179`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `5.5`, LSTM `0.6579`, XGBoost `0.5293`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.0`, LSTM `0.6511`, XGBoost `0.5293`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `73.0`, LSTM `0.4395`, XGBoost `0.3191`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.4365`, XGBoost `0.3176`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.6407`, XGBoost `0.5293`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `71.5`, LSTM `0.4276`, XGBoost `0.3176`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `3.5`, LSTM `0.6427`, XGBoost `0.5333`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `3.0`, LSTM `0.6422`, XGBoost `0.5333`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `2.5`, LSTM `0.6413`, XGBoost `0.5363`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
