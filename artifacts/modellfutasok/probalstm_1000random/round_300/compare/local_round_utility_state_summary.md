# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `14`
- rows: `175`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 175 | 1.000 | 0.688994 | 0.763016 | -0.074022 | 0 | 175 | 0.988571 | 1.000000 |
| active/recent utility | 175 | 1.000 | 0.688994 | 0.763016 | -0.074022 | 0 | 175 | 0.988571 | 1.000000 |
| strong utility action | 122 | 0.697 | 0.613341 | 0.705579 | -0.092238 | 0 | 122 | 0.983607 | 1.000000 |
| utility damage | 10 | 0.057 | 0.589488 | 0.638347 | -0.048859 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 122 | 0.697 | 0.613341 | 0.705579 | -0.092238 | 0 | 122 | 0.983607 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 175 | 1.000 | 0.688994 | 0.763016 | -0.074022 | 0 | 175 | 0.988571 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `68.5s`, rows `122`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.5`, LSTM `0.5562`, XGBoost `0.7645`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.5571`, XGBoost `0.7645`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.5588`, XGBoost `0.7645`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.5615`, XGBoost `0.7641`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5923`, XGBoost `0.7943`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.5654`, XGBoost `0.7645`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.5688`, XGBoost `0.7648`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.5769`, XGBoost `0.7658`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.5806`, XGBoost `0.7689`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.5814`, XGBoost `0.7689`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
