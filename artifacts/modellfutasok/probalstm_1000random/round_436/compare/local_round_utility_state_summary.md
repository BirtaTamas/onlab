# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `12`
- rows: `156`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 156 | 1.000 | 0.662012 | 0.635257 | 0.026755 | 122 | 34 | 1.000000 | 1.000000 |
| active/recent utility | 156 | 1.000 | 0.662012 | 0.635257 | 0.026755 | 122 | 34 | 1.000000 | 1.000000 |
| strong utility action | 135 | 0.865 | 0.665328 | 0.645785 | 0.019543 | 101 | 34 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 135 | 0.865 | 0.665328 | 0.645785 | 0.019543 | 101 | 34 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 156 | 1.000 | 0.662012 | 0.635257 | 0.026755 | 122 | 34 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `77.5s`, rows `135`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.0`, LSTM `0.5982`, XGBoost `0.5085`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.5980`, XGBoost `0.5099`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6437`, XGBoost `0.5609`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6404`, XGBoost `0.5606`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.5813`, XGBoost `0.5051`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6378`, XGBoost `0.5628`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5750`, XGBoost `0.5014`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.6244`, XGBoost `0.5587`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.5863`, XGBoost `0.5246`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.5862`, XGBoost `0.5246`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
