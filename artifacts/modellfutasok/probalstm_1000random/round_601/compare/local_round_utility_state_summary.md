# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `4`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.774827 | 0.754370 | 0.020457 | 105 | 125 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.774827 | 0.754370 | 0.020457 | 105 | 125 | 1.000000 | 1.000000 |
| strong utility action | 160 | 0.696 | 0.735196 | 0.705814 | 0.029382 | 90 | 70 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.087 | 0.676123 | 0.638867 | 0.037256 | 11 | 9 | 1.000000 | 1.000000 |
| active smoke/inferno | 160 | 0.696 | 0.735196 | 0.705814 | 0.029382 | 90 | 70 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.774827 | 0.754370 | 0.020457 | 105 | 125 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `64.0s`, rows `116`
- `76.5s` - `98.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `21.0`, LSTM `0.6532`, XGBoost `0.5101`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.6449`, XGBoost `0.5113`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.6338`, XGBoost `0.5029`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `3.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.6241`, XGBoost `0.5021`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.6226`, XGBoost `0.5040`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `3.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6658`, XGBoost `0.5481`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6205`, XGBoost `0.5029`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `3.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.6205`, XGBoost `0.5036`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `3.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.6271`, XGBoost `0.5108`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.6169`, XGBoost `0.5021`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
