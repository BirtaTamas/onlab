# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv`
- round_num: `7`
- rows: `293`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 293 | 1.000 | 0.330238 | 0.477010 | -0.146772 | 256 | 37 | 0.665529 | 0.296928 |
| active/recent utility | 293 | 1.000 | 0.330238 | 0.477010 | -0.146772 | 256 | 37 | 0.665529 | 0.296928 |
| strong utility action | 184 | 0.628 | 0.436528 | 0.592661 | -0.156133 | 168 | 16 | 0.494565 | 0.108696 |
| utility damage | 10 | 0.034 | 0.633758 | 0.733984 | -0.100225 | 10 | 0 | 0.200000 | 0.000000 |
| active smoke/inferno | 173 | 0.590 | 0.421310 | 0.588154 | -0.166845 | 166 | 7 | 0.526012 | 0.115607 |
| recent utility last 5s | 21 | 0.072 | 0.408462 | 0.402610 | 0.005853 | 7 | 14 | 0.476190 | 0.476190 |
| flash effect present | 293 | 1.000 | 0.330238 | 0.477010 | -0.146772 | 256 | 37 | 0.665529 | 0.296928 |

## Active Smoke/Inferno Intervals

- `8.0s` - `54.5s`, rows `94`
- `73.5s` - `112.5s`, rows `79`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `77.0`, LSTM `0.2322`, XGBoost `0.6538`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.2323`, XGBoost `0.6538`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.2371`, XGBoost `0.6538`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.2392`, XGBoost `0.6538`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.2497`, XGBoost `0.6538`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.2509`, XGBoost `0.6538`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.2575`, XGBoost `0.6538`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.2687`, XGBoost `0.6538`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.2730`, XGBoost `0.6538`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.2940`, XGBoost `0.6538`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
