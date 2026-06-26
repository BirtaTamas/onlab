# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-faze-bo3-q02I_n27c_oaVV09Kplodn/mouz-vs-faze-m2-mirage.csv`
- round_num: `11`
- rows: `144`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.309897 | 0.386654 | -0.076757 | 131 | 13 | 0.791667 | 0.500000 |
| active/recent utility | 144 | 1.000 | 0.309897 | 0.386654 | -0.076757 | 131 | 13 | 0.791667 | 0.500000 |
| strong utility action | 130 | 0.903 | 0.290952 | 0.370663 | -0.079710 | 117 | 13 | 0.769231 | 0.553846 |
| utility damage | 26 | 0.181 | 0.349744 | 0.487470 | -0.137725 | 26 | 0 | 0.846154 | 0.384615 |
| active smoke/inferno | 130 | 0.903 | 0.290952 | 0.370663 | -0.079710 | 117 | 13 | 0.769231 | 0.553846 |
| recent utility last 5s | 10 | 0.069 | 0.753590 | 0.720431 | 0.033159 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 144 | 1.000 | 0.309897 | 0.386654 | -0.076757 | 131 | 13 | 0.791667 | 0.500000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `71.5s`, rows `130`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `39.5`, LSTM `0.0636`, XGBoost `0.4141`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.1133`, XGBoost `0.4319`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `28.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.0940`, XGBoost `0.4031`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `28.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.0947`, XGBoost `0.4038`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `28.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.1046`, XGBoost `0.4068`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `28.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.1052`, XGBoost `0.4068`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `28.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.1042`, XGBoost `0.4031`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `28.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.0982`, XGBoost `0.3947`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `21.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1936`, XGBoost `0.4626`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.0327`, XGBoost `0.2998`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
