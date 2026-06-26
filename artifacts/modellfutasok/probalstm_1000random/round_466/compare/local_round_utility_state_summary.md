# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `21`
- rows: `183`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 183 | 1.000 | 0.746988 | 0.794241 | -0.047253 | 46 | 137 | 0.978142 | 0.961749 |
| active/recent utility | 183 | 1.000 | 0.746988 | 0.794241 | -0.047253 | 46 | 137 | 0.978142 | 0.961749 |
| strong utility action | 94 | 0.514 | 0.774208 | 0.843622 | -0.069414 | 13 | 81 | 1.000000 | 1.000000 |
| utility damage | 16 | 0.087 | 0.711861 | 0.709382 | 0.002480 | 9 | 7 | 1.000000 | 1.000000 |
| active smoke/inferno | 88 | 0.481 | 0.766840 | 0.836492 | -0.069652 | 13 | 75 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 183 | 1.000 | 0.746988 | 0.794241 | -0.047253 | 46 | 137 | 0.978142 | 0.961749 |

## Active Smoke/Inferno Intervals

- `9.0s` - `52.5s`, rows `88`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.7697`, XGBoost `0.8999`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.7732`, XGBoost `0.9000`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.7610`, XGBoost `0.8860`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.7785`, XGBoost `0.9000`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.7653`, XGBoost `0.8860`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.7630`, XGBoost `0.8825`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.7696`, XGBoost `0.8867`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.7830`, XGBoost `0.8994`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.7831`, XGBoost `0.8994`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.6769`, XGBoost `0.7931`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `46.0`, recent_utility `0`
